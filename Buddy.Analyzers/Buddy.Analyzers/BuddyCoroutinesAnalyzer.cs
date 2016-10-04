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
            List<MethodDeclarationSyntax> methods = comp.SyntaxTrees.SelectMany(
                                                            st =>
                                                                st.GetRoot()
                                                                  .DescendantNodes()
                                                                  .OfType
                                                                  <MethodDeclarationSyntax>())
                                                        .ToList();

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

            // Seed with declared async methods.
            foreach (MethodDeclarationSyntax method in methods)
            {
                var semanticModel = comp.GetSemanticModel(method.SyntaxTree);
                IMethodSymbol sym = semanticModel.GetDeclaredSymbol(method);
                if (!ctx.IsAsyncCompatible(sym) ||
                    method.Modifiers.All(st => st.Kind() != SyntaxKind.AsyncKeyword))
                    continue;

                ctx.AddOrGetNode(sym, method);
            }

            // Expand seed set to full call graph.
            foreach (
                KeyValuePair<IMethodSymbol, MethodNode> kvp in ctx.Methods.ToList())
            {
                SemanticModel model = comp.GetSemanticModel(kvp.Value.AsyncDeclaration.SyntaxTree);
                if (kvp.Value.AsyncDeclaration.Body == null)
                    continue;

                List<InvocationExpressionSyntax> invocations =
                    kvp.Value.AsyncDeclaration.Body.DescendantNodes(
                           sn => !(sn is LambdaExpressionSyntax))
                       .OfType<InvocationExpressionSyntax>().ToList();

                foreach (InvocationExpressionSyntax invoc in invocations)
                {
                    IMethodSymbol calledSym = model.GetSymbolInfo(invoc).Symbol as IMethodSymbol;
                    if (calledSym == null)
                        continue;

                    if (!ctx.IsAsyncCompatible(calledSym))
                        continue;

                    MethodNode called = ctx.AddOrGetNode(calledSym, null);

                    bool isAwaited = IsAwaited(invoc);
                    called.Callers.Add(new Call(kvp.Value, isAwaited));
                    kvp.Value.Callees.Add(new Call(called, isAwaited));
                }
            }

            // Mark commonly known coroutines and external tasks.
            string[] bosslandPrefixes =
            {
                "Buddy.Coroutines.",
                "Styx.CommonBot.Coroutines."
            };

            foreach (
                MethodNode method in
                ctx.MethodsByName.Where(kvp => bosslandPrefixes.Any(p => kvp.Key.StartsWith(p)))
                   .SelectMany(kvp => kvp.Value))
                ctx.Mark(method, AsyncMethodKind.Coroutine);

            foreach (
                MethodNode method in
                ctx.MethodsByName.Where(kvp => kvp.Key.StartsWith("System."))
                   .SelectMany(kvp => kvp.Value))
                ctx.Mark(method, AsyncMethodKind.ExternalTask);

            foreach (MethodNode method in 
                ctx.MethodsByName.Where(kvp => kvp.Key.EndsWith("Coroutine"))
                   .SelectMany(kvp => kvp.Value))
                ctx.Mark(method, AsyncMethodKind.Coroutine);

            foreach (MethodNode method in 
                ctx.MethodsByName.Where(kvp => kvp.Key.EndsWith("Async"))
                   .SelectMany(kvp => kvp.Value))
                ctx.Mark(method, AsyncMethodKind.ExternalTask);

            ctx.MarkSingle("System.Threading.Task.CompletedTask", AsyncMethodKind.Both);
            ctx.MarkSingle("System.Threading.Task.FromResult", AsyncMethodKind.Both);

            // Generate errors.
            foreach (
                KeyValuePair<IMethodSymbol, MethodNode> kvp in
                ctx.Methods.Where(
                       kvp =>
                           kvp.Value.AsyncDeclaration != null &&
                           (kvp.Value.Kind == AsyncMethodKind.Coroutine ||
                            kvp.Value.Kind == AsyncMethodKind.ExternalTask)))
            {
                SyntaxTree tree = kvp.Value.AsyncDeclaration.SyntaxTree;
                SemanticModel model = comp.GetSemanticModel(tree);

                IEnumerable<SyntaxNode> invocs =
                    kvp.Value.AsyncDeclaration
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
                        kvp.Value.AsyncDeclaration
                           .DescendantNodes(sn => !(sn is LambdaExpressionSyntax))
                           .OfType<TryStatementSyntax>();

                    foreach (TryStatementSyntax tryStatement in tryStatements)
                    {
                        bool tryContainsAwait =
                            tryStatement.Block
                                        .DescendantNodes(sn => !(sn is LambdaExpressionSyntax))
                                        .Any(sn => sn is AwaitExpressionSyntax);

                        if (tryContainsAwait)
                        {
                            foreach (CatchClauseSyntax catchClause in tryStatement.Catches)
                            {
                                if (ctx.CatchClauseCatchesCoroutineStopped(catchClause, model))
                                {
                                    context.ReportDiagnostic(
                                        Diagnostic.Create(s_doNotCatchCoroutineStoppedExceptionRule,
                                                          catchClause.GetLocation()));
                                }
                            }

                            if (tryStatement.Finally != null)
                            {
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
            }
        }

        private static bool IsAwaited(SyntaxNode node)
        {
            return node.Parent is AwaitExpressionSyntax;
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

            // Marks a node and all its awaited/awaiting ancestors/descendants as the specified kind.
            public void Mark(MethodNode node, AsyncMethodKind kind)
            {
                node.Kind = kind;

                foreach (Call parentCall in node.Callers)
                {
                    // Only mark coroutines if awaited
                    if (parentCall.Other.Kind == AsyncMethodKind.Unknown &&
                        (kind != AsyncMethodKind.Coroutine || parentCall.IsAwaited))
                        Mark(parentCall.Other, kind);
                }

                foreach (Call childCall in node.Callees)
                {
                    if (childCall.Other.Kind == AsyncMethodKind.Unknown &&
                        (kind != AsyncMethodKind.Coroutine || childCall.IsAwaited))
                        Mark(childCall.Other, kind);
                }
            }

            public void MarkSingle(string name, AsyncMethodKind kind)
            {
                List<MethodNode> list;
                if (!MethodsByName.TryGetValue(name, out list))
                    return;

                foreach (MethodNode node in list)
                    node.Kind = kind;
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
                                                           SemanticModel model)
            {
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
                        return false;
                }

                ControlFlowAnalysis controlFlowResult =
                    model.AnalyzeControlFlow(catchClause.Block);
                if (!controlFlowResult.Succeeded)
                    return false;

                // Check if this catch always rethrows
                if (!controlFlowResult.EndPointIsReachable &&
                    controlFlowResult.ExitPoints.Length == 0 &&
                    controlFlowResult.ReturnStatements.Length == 0)
                    return false;

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

                if (clause.Block != null &&
                    clause.Block.Statements.Count > 0)
                {
                    // Check pattern "if (ex is CoroutineStoppedException) throw;"
                    IfStatementSyntax @if = clause.Block.Statements[0] as IfStatementSyntax;
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
                }

                return false;
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
            public MethodNode(IMethodSymbol method, MethodDeclarationSyntax asyncDeclaration)
            {
                Method = method;
                AsyncDeclaration = asyncDeclaration;
            }

            public IMethodSymbol Method { get; }
            public HashSet<Call> Callers { get; } = new HashSet<Call>();
            public HashSet<Call> Callees { get; } = new HashSet<Call>();
            public AsyncMethodKind Kind { get; set; } = AsyncMethodKind.Unknown;
            public MethodDeclarationSyntax AsyncDeclaration { get; }

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
            public Call(MethodNode other, bool isAwaited)
            {
                Other = other;
                IsAwaited = isAwaited;
            }

            public MethodNode Other { get; }
            public bool IsAwaited { get; }

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
