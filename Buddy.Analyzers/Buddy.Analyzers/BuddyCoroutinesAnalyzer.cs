using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
using CSharpExtensions = Microsoft.CodeAnalysis.CSharp.CSharpExtensions;
using System.IO;

namespace Buddy.Analyzers
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public class BuddyCoroutinesAnalyzer : DiagnosticAnalyzer
    {
        private static readonly DiagnosticDescriptor s_awaitAllCoroutinesImmediatelyRule =
            new DiagnosticDescriptor(
                "BDY0001",
                "BDY0001: Coroutines should be awaited immediately",
                "The coroutine '{0}' should be awaited immediately",
                "Coroutines",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true);

        private static readonly DiagnosticDescriptor s_doNotAwaitExternalTaskFromCoroutineRule =
            new DiagnosticDescriptor(
                "BDY0002",
                "BDY0002: External tasks should not be awaited from coroutines",
                "The external task '{0}' should not be awaited from coroutine '{1}'",
                "Coroutines",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true);

        private static readonly DiagnosticDescriptor s_doNotUseCoroutinesFromExternalTaskRule =
            new DiagnosticDescriptor(
                "BDY0003",
                "BDY0003: Coroutines should not be used from external tasks",
                "The coroutine '{0}' should not be used from external task '{1}'",
                "Coroutines",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true);

        private static readonly DiagnosticDescriptor s_doNotCatchCoroutineStoppedExceptionRule =
            new DiagnosticDescriptor(
                "BDY0004",
                "BDY0004: Do not catch Buddy.Coroutines.CoroutineStoppedException",
                "Exception handler catches Buddy.Coroutines.CoroutineStoppedException",
                "Coroutines",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true);

        private static readonly DiagnosticDescriptor s_doNotAwaitInsideFinallyRule =
            new DiagnosticDescriptor(
                "BDY0005",
                "BDY0005: Do not await inside finally blocks when try block contains an await",
                "finally-block should not await when try-block contains an await",
                "Coroutines",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true);

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
            =>
            ImmutableArray.Create(s_awaitAllCoroutinesImmediatelyRule,
                                  s_doNotAwaitExternalTaskFromCoroutineRule,
                                  s_doNotUseCoroutinesFromExternalTaskRule,
                                  s_doNotCatchCoroutineStoppedExceptionRule,
                                  s_doNotAwaitInsideFinallyRule);

        public override void Initialize(AnalysisContext context)
        {
            context.RegisterCompilationAction(ctx =>
                                              {
                                                  try
                                                  {
                                                      AnalyzeCompilation(ctx);
                                                  }
                                                  catch (Exception ex)
                                                  {
                                                      throw new Exception(ex.ToString().Replace(Environment.NewLine, "  "));
                                                  }
                                              });
        }

        private static void AnalyzeCompilation(CompilationAnalysisContext context)
        {
            Compilation comp = context.Compilation;

            AnalyzeContext ctx = new AnalyzeContext
                                 {
                                     TaskType = comp.GetTypeByMetadataName(typeof(Task).FullName),
                                     TaskTType = comp.GetTypeByMetadataName(typeof(Task<>).FullName),
                                     ExceptionType =
                                         comp.GetTypeByMetadataName(typeof(Exception).FullName),
                                     CoroutineType =
                                         comp.GetTypeByMetadataName("Buddy.Coroutines.Coroutine"),
                                     CoroutineStoppedExceptionType =
                                         comp.GetTypeByMetadataName(
                                             "Buddy.Coroutines.CoroutineStoppedException"),
                                 };

            if (ctx.TaskType == null ||
                ctx.TaskTType == null ||
                ctx.CoroutineType == null ||
                ctx.ExceptionType == null ||
                ctx.CoroutineStoppedExceptionType == null)
                return;

            // Seed with declared task methods.
            FindDeclaredTasks(ctx, comp);

            // Expand seed set to full call graph.
            GenerateCallGraph(ctx, comp);

            // Mark task methods as coroutines, external tasks or both.
            ClassifyTasks(ctx);

            // Generate errors.
            foreach (
                KeyValuePair<IMethodSymbol, MethodNode> kvp in
                ctx.Methods.Where(
                       kvp =>
                           kvp.Value.Declaration != null &&
                           kvp.Value.IsAsync.HasValue &&
                           kvp.Value.IsAsync.Value &&
                           (kvp.Value.Kind == AsyncMethodKind.Coroutine ||
                            kvp.Value.Kind == AsyncMethodKind.ExternalTask)))
            {
                SyntaxTree tree = kvp.Value.Declaration.SyntaxTree;
                SemanticModel model = comp.GetSemanticModel(tree);

                IEnumerable<SyntaxNode> invocs =
                    kvp.Value.Declaration
                       .DescendantNodes(sn => !(sn is LambdaExpressionSyntax))
                       .OfType<InvocationExpressionSyntax>();

                bool callerIsCoroutine = kvp.Value.Kind == AsyncMethodKind.Coroutine;
                bool callerIsExternalTask = kvp.Value.Kind == AsyncMethodKind.ExternalTask;

                foreach (SyntaxNode invoc in invocs)
                {
                    IMethodSymbol called = model.GetSymbolInfo(invoc).Symbol as IMethodSymbol;
                    if (called == null)
                        continue;

                    if (!ctx.IsAsyncCompatible(called))
                        continue;

                    MethodNode calledNode = ctx.GetNode(called);
                    if (calledNode == null)
                        continue;

                    bool calledIsCoroutine = calledNode.Kind == AsyncMethodKind.Coroutine;
                    bool calledIsExternalTask = calledNode.Kind == AsyncMethodKind.ExternalTask;

                    if (callerIsCoroutine && calledIsCoroutine)
                    {
                        if (!IsAwaited(invoc))
                        {
                            context.ReportDiagnostic(
                                Diagnostic.Create(s_awaitAllCoroutinesImmediatelyRule,
                                                  invoc.GetLocation(),
                                                  GetFullMethodName(called)));
                        }
                    }
                    else if (callerIsCoroutine && calledIsExternalTask)
                    {
                        if (IsAwaited(invoc))
                        {
                            context.ReportDiagnostic(
                                Diagnostic.Create(s_doNotAwaitExternalTaskFromCoroutineRule,
                                                  invoc.GetLocation(),
                                                  GetFullMethodName(called),
                                                  GetFullMethodName(kvp.Key)));
                        }
                    }
                    else if (callerIsExternalTask && calledIsCoroutine)
                    {
                        context.ReportDiagnostic(
                            Diagnostic.Create(s_doNotUseCoroutinesFromExternalTaskRule,
                                              invoc.GetLocation(),
                                              GetFullMethodName(called),
                                              GetFullMethodName(kvp.Key)));
                    }
                }

                if (callerIsCoroutine)
                {
                    IEnumerable<TryStatementSyntax> tryStatements =
                        kvp.Value.Declaration
                           .DescendantNodes(sn => !(sn is LambdaExpressionSyntax))
                           .OfType<TryStatementSyntax>();

                    foreach (TryStatementSyntax tryStatement in tryStatements)
                    {
                        bool tryContainsAwait =
                            tryStatement.Block
                                        .DescendantNodes(sn => !(sn is LambdaExpressionSyntax))
                                        .Any(sn => sn is AwaitExpressionSyntax);

                        if (!tryContainsAwait)
                            continue;

                        // Check if it catches coroutine stopped exception...
                        foreach (CatchClauseSyntax catchClause in tryStatement.Catches)
                        {
                            bool rethrows;
                            if (!ctx.CatchClauseCatchesCoroutineStopped(
                                catchClause, model, out rethrows))
                                continue;

                            if (!rethrows)
                            {
                                context.ReportDiagnostic(
                                    Diagnostic.Create(
                                        s_doNotCatchCoroutineStoppedExceptionRule,
                                        catchClause.GetLocation()));
                            }

                            break;
                        }

                        if (tryStatement.Finally == null)
                            continue;

                        // Check if it awaits inside finally
                        IEnumerable<AwaitExpressionSyntax> awaits =
                            tryStatement.Finally.Block.DescendantNodes(
                                            sn => !(sn is LambdaExpressionSyntax))
                                        .OfType<AwaitExpressionSyntax>();

                        foreach (AwaitExpressionSyntax @await in awaits)
                        {
                            context.ReportDiagnostic(
                                Diagnostic.Create(s_doNotAwaitInsideFinallyRule,
                                                  @await.GetLocation()));
                        }
                    }
                }
            }
        }

        private static void FindDeclaredTasks(AnalyzeContext ctx, Compilation comp)
        {
            IEnumerable<MethodDeclarationSyntax> methods =
                comp.SyntaxTrees.SelectMany(
                        st => st.GetRoot().DescendantNodes().OfType<MethodDeclarationSyntax>());

            foreach (MethodDeclarationSyntax method in methods)
            {
                SemanticModel semanticModel = comp.GetSemanticModel(method.SyntaxTree);
                IMethodSymbol sym = semanticModel.GetDeclaredSymbol(method);
                if (method.Body == null ||
                    !ctx.IsAsyncCompatible(sym))
                    continue;

                ctx.AddOrGetNode(sym, method);
            }
        }

        private static void GenerateCallGraph(AnalyzeContext ctx, Compilation comp)
        {
            foreach (MethodNode method in ctx.Methods.Values.ToList())
            {
                SemanticModel model = comp.GetSemanticModel(method.Declaration.SyntaxTree);

                IEnumerable<InvocationExpressionSyntax> invocations =
                    method.Declaration.Body.DescendantNodes(
                              sn => !(sn is LambdaExpressionSyntax))
                          .OfType<InvocationExpressionSyntax>();

                foreach (InvocationExpressionSyntax invoc in invocations)
                {
                    IMethodSymbol calledSym = model.GetSymbolInfo(invoc).Symbol as IMethodSymbol;
                    if (calledSym == null)
                        continue;

                    if (!ctx.IsAsyncCompatible(calledSym))
                        continue;

                    MethodNode called = ctx.AddOrGetNode(calledSym, null);

                    bool isAwaited = IsAwaited(invoc);
                    bool isReturned = IsReturned(invoc);
                    called.Callers.Add(new Call(method, isAwaited, isReturned));
                    method.Callees.Add(new Call(called, isAwaited, isReturned));
                }
            }
        }

        private static void ClassifyTasks(AnalyzeContext ctx)
        {
            // Pass 1: Mark our assumptions.
            string[] bosslandPrefixes =
            {
                "Buddy.Coroutines.",
                "Styx.CommonBot.Coroutines."
            };

            foreach (
                MethodNode method in ctx.GetByName(name => bosslandPrefixes.Any(name.StartsWith)))
            {
                method.Kind = AsyncMethodKind.Coroutine;
            }

            foreach (MethodNode method in ctx.GetByName(name => name.EndsWith("Coroutine")))
                method.Kind = AsyncMethodKind.Coroutine;

            foreach (MethodNode method in ctx.GetByName(name => name.StartsWith("System.")))
                method.Kind = AsyncMethodKind.ExternalTask;

            foreach (MethodNode method in ctx.GetByName(name => name.EndsWith("Async")))
                method.Kind = AsyncMethodKind.ExternalTask;

            string[] bothKindPrefixes =
            {
                "System.Threading.Tasks.Task.CompletedTask",
                "System.Threading.Tasks.Task.FromResult",
                "System.Func"
            };

            foreach (MethodNode method in ctx.GetByName(name => bothKindPrefixes.Any(name.StartsWith)))
                method.Kind = AsyncMethodKind.Both;

            // Pass 2: Classify recursively.
            ctx.FloodClassify(AsyncMethodKind.Coroutine);
            ctx.FloodClassify(AsyncMethodKind.ExternalTask);
        }

        private static bool IsAwaited(SyntaxNode node)
        {
            return node.Parent is AwaitExpressionSyntax;
        }

        private static bool IsReturned(SyntaxNode node)
        {
            return node.Parent is ReturnStatementSyntax;
        }

        private static string GetFullMethodName(IMethodSymbol symbol)
        {
            ISymbol s = symbol.ContainingType;
            var sb = new StringBuilder(s.MetadataName);

            ISymbol last = s;
            s = s.ContainingSymbol;
            while (!IsRootNamespace(s))
            {
                if (s is ITypeSymbol && last is ITypeSymbol)
                {
                    sb.Insert(0, '+');
                }
                else
                {
                    sb.Insert(0, '.');
                }
                sb.Insert(0, s.MetadataName);

                last = s;
                s = s.ContainingSymbol;
            }

            return sb + "." + symbol.Name;
        }

        private static bool IsRootNamespace(ISymbol s)
        {
            return s is INamespaceSymbol && ((INamespaceSymbol)s).IsGlobalNamespace;
        }


        private class AnalyzeContext
        {
            public INamedTypeSymbol TaskType { get; set; }
            public INamedTypeSymbol TaskTType { get; set; }
            public INamedTypeSymbol CoroutineType { get; set; }
            public INamedTypeSymbol ExceptionType { get; set; }
            public INamedTypeSymbol CoroutineStoppedExceptionType { get; set; }

            public Dictionary<IMethodSymbol, MethodNode> Methods { get; } =
                new Dictionary<IMethodSymbol, MethodNode>();

            public Dictionary<string, List<MethodNode>> MethodsByName { get; } =
                new Dictionary<string, List<MethodNode>>();

            public MethodNode GetNode(IMethodSymbol symbol)
            {
                MethodNode node;
                return Methods.TryGetValue(symbol, out node) ? node : null;
            }

            public MethodNode AddOrGetNode(IMethodSymbol symbol, MethodDeclarationSyntax asyncDecl)
            {
                MethodNode node;
                if (Methods.TryGetValue(symbol, out node))
                    return node;

                node = new MethodNode(symbol, asyncDecl);
                Methods.Add(symbol, node);

                string name = GetFullMethodName(symbol);
                List<MethodNode> list;
                if (!MethodsByName.TryGetValue(name, out list))
                    MethodsByName[name] = list = new List<MethodNode>();

                list.Add(node);
                return node;
            }

            public IEnumerable<MethodNode> GetByName(Func<string, bool> matches)
            {
                return MethodsByName.Where(kvp => matches(kvp.Key)).SelectMany(kvp => kvp.Value);
            }

            /// <summary>
            /// Recursively classifies all parents/children of asyncs of the specified
            /// kind.
            /// </summary>
            /// <param name="kind"></param>
            public void FloodClassify(AsyncMethodKind kind)
            {
                foreach (MethodNode method in Methods.Values)
                {
                    if (method.Kind == kind)
                        ClassifyRecursively(method);
                }
            }

            private void ClassifyRecursively(MethodNode node)
            {
                // For coroutines we have a special case. When a coroutine uses
                // a task without awaiting it, we cannot mark it as a coroutine,
                // as it could be external.
                foreach (Call parentCall in node.Callers)
                {
                    if (parentCall.Other.Kind != AsyncMethodKind.Unknown)
                        continue;

                    // Parent of external task is only an external task if awaited
                    // or returned, since a coroutine can use an external task.
                    if (node.Kind == AsyncMethodKind.ExternalTask &&
                        !parentCall.IsAwaited &&
                        !parentCall.IsReturned)
                        continue;

                    parentCall.Other.Kind = node.Kind;
                    ClassifyRecursively(parentCall.Other);
                }

                foreach (Call childCall in node.Callees)
                {
                    if (childCall.Other.Kind != AsyncMethodKind.Unknown)
                        continue;

                    // Child of coroutine is only coroutine if it is awaited/returned
                    // - otherwise it could be an external task.
                    if (node.Kind == AsyncMethodKind.Coroutine &&
                        !childCall.IsAwaited &&
                        !childCall.IsReturned)
                        continue;

                    childCall.Other.Kind = node.Kind;
                    ClassifyRecursively(childCall.Other);
                }
            }

            public bool IsAsyncCompatible(IMethodSymbol symbol)
            {
                INamedTypeSymbol retType = symbol.ReturnType as INamedTypeSymbol;
                if (retType == null)
                    return false;

                if (retType.Equals(TaskType))
                    return true;

                if (!retType.IsGenericType)
                    return false;

                return retType.OriginalDefinition.Equals(TaskTType);
            }

            public bool CatchClauseCatchesCoroutineStopped(CatchClauseSyntax catchClause,
                                                           SemanticModel model, out bool rethrows)
            {
                rethrows = false;
                if (catchClause.Declaration != null)
                {
                    INamedTypeSymbol exceptionSymbol =
                        model.GetSymbolInfo(catchClause.Declaration.Type).Symbol as
                            INamedTypeSymbol;

                    if (exceptionSymbol == null)
                        return false;

                    if (exceptionSymbol != ExceptionType &&
                        exceptionSymbol != CoroutineStoppedExceptionType)
                        return false;

                    if (RethrowsCoroutineStoppedException(catchClause, model))
                        rethrows = true;
                }

                if (RethrowsAnything(catchClause))
                    rethrows = true;

                return true;
            }

            private bool RethrowsCoroutineStoppedException(CatchClauseSyntax clause,
                                                           SemanticModel model)
            {
                // Simple pattern matching here. I don't think we can do better...
                ILocalSymbol exceptionSymbol =
                    model.GetDeclaredSymbol(clause.Declaration);

                if (clause.Filter != null)
                {
                    // Check pattern !(ex is CoroutineStoppedException)

                    ExpressionSyntax exp = clause.Filter.FilterExpression;
                    PrefixUnaryExpressionSyntax unExp = exp as PrefixUnaryExpressionSyntax;
                    if (unExp != null && unExp.OperatorToken.Kind() == SyntaxKind.ExclamationToken)
                    {
                        ParenthesizedExpressionSyntax parens =
                            unExp.Operand as ParenthesizedExpressionSyntax;
                        var exp2 = parens?.Expression;
                        if (IsExIsCoroutineStoppedExceptionExp(exp2, exceptionSymbol, model))
                            return true;
                    }
                }

                if (clause.Block != null)
                {
                    // Check pattern "if (ex is CoroutineStoppedException) throw;"
                    foreach (StatementSyntax statement in clause.Block.Statements)
                    {
                        IfStatementSyntax @if = statement as IfStatementSyntax;
                        if (@if != null &&
                            IsExIsCoroutineStoppedExceptionExp(@if.Condition, exceptionSymbol, model))
                        {
                            if (@if.Statement is ThrowStatementSyntax)
                                return true;

                            BlockSyntax block = @if.Statement as BlockSyntax;
                            if (block != null && block.Statements.Count > 0 &&
                                block.Statements[0] is ThrowStatementSyntax)
                                return true;
                        }

                        if (CanThrow(statement))
                            break;
                    }
                }

                return false;
            }

            private bool RethrowsAnything(CatchClauseSyntax catchClause)
            {
                if (catchClause.Block == null)
                    return false;

                foreach (StatementSyntax statement in catchClause.Block.Statements)
                {
                    ThrowStatementSyntax throwStatement = statement as ThrowStatementSyntax;
                    if (throwStatement != null && throwStatement.Expression == null)
                        return true;

                    if (CanThrow(statement))
                        break;
                }

                return false;
            }

            private bool CanThrow(StatementSyntax statement)
            {
                // Check for simple assignments for now.
                ExpressionStatementSyntax expStatement = statement as ExpressionStatementSyntax;
                AssignmentExpressionSyntax exp = expStatement?.Expression as AssignmentExpressionSyntax;
                if (expStatement == null || exp == null)
                    return true;

                if (exp.Left is IdentifierNameSyntax &&
                    (exp.Right is LiteralExpressionSyntax || exp.Right is IdentifierNameSyntax))
                    return false;

                return true;
            }

            private bool IsExIsCoroutineStoppedExceptionExp(ExpressionSyntax exp,
                                                            ILocalSymbol exceptionSymbol,
                                                            SemanticModel model)
            {
                BinaryExpressionSyntax binExp = exp as BinaryExpressionSyntax;
                if (binExp == null || binExp.OperatorToken.Kind() != SyntaxKind.IsKeyword)
                    return false;

                ILocalSymbol leftSymbol = model.GetSymbolInfo(binExp.Left).Symbol as ILocalSymbol;
                INamedTypeSymbol rightSymbol =
                    model.GetSymbolInfo(binExp.Right).Symbol as INamedTypeSymbol;

                if (leftSymbol != null && leftSymbol.Equals(exceptionSymbol) &&
                    rightSymbol == CoroutineStoppedExceptionType)
                    return true;
                return false;
            }
        }

        private class MethodNode
        {
            public MethodNode(IMethodSymbol method, MethodDeclarationSyntax declaration)
            {
                Method = method;
                Declaration = declaration;
                if (declaration != null)
                {
                    IsAsync =
                        declaration.Modifiers.Any(m => m.Kind() == SyntaxKind.AsyncKeyword);
                }
            }

            public IMethodSymbol Method { get; }
            public HashSet<Call> Callers { get; } = new HashSet<Call>();
            public HashSet<Call> Callees { get; } = new HashSet<Call>();
            public AsyncMethodKind Kind { get; set; } = AsyncMethodKind.Unknown;
            public MethodDeclarationSyntax Declaration { get; }
            public bool? IsAsync { get; }

            protected bool Equals(MethodNode other)
            {
                return Method.Equals(other.Method);
            }

            public override bool Equals(object obj)
            {
                if (ReferenceEquals(null, obj))
                    return false;
                if (ReferenceEquals(this, obj))
                    return true;
                if (obj.GetType() != GetType())
                    return false;
                return Equals((MethodNode)obj);
            }

            public override int GetHashCode()
            {
                return Method.GetHashCode();
            }

            public static bool operator ==(MethodNode left, MethodNode right)
            {
                return Equals(left, right);
            }

            public static bool operator !=(MethodNode left, MethodNode right)
            {
                return !Equals(left, right);
            }
        }

        private struct Call
        {
            public Call(MethodNode other, bool isAwaited, bool isReturned)
            {
                Other = other;
                IsAwaited = isAwaited;
                IsReturned = isReturned;
            }

            public MethodNode Other { get; }
            public bool IsAwaited { get; }
            public bool IsReturned { get; }

            public bool Equals(Call other)
            {
                return Other.Equals(other.Other);
            }

            public override bool Equals(object obj)
            {
                if (ReferenceEquals(null, obj))
                    return false;
                return obj is Call && Equals((Call)obj);
            }

            public override int GetHashCode()
            {
                return Other.GetHashCode();
            }

            public static bool operator ==(Call left, Call right)
            {
                return left.Equals(right);
            }

            public static bool operator !=(Call left, Call right)
            {
                return !left.Equals(right);
            }
        }

        private enum AsyncMethodKind
        {
            Unknown,
            Coroutine,
            ExternalTask,
            Both,
        }
    }
}
