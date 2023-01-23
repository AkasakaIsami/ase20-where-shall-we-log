package visitor;

import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import util.LogUtil;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * <p>You can check the problem detail on <a href="">Leetcode</a>.</p>
 *
 * @author Akasaka Isami
 * @since 2023/1/17 10:40 PM
 */
public class MethodVisitor extends VoidVisitorAdapter<String> {
    // 类名@函数名@行号：该语句的seq
    public static Map<String, String> posSeq = new HashMap<>();
    public static Map<String, String> negSeq = new HashMap<>();

    String key;
    // 对于每一个函数 都要维护一个记录遍历到当前哪个节点的sb
    StringBuilder trace = new StringBuilder();


    @Override
    public void visit(MethodDeclaration node, String fileName) {
        if (node != null && node.isMethodDeclaration()) {
            System.out.println("遍历函数" + node.getNameAsString() + "中");
            MethodDeclaration methodDeclaration = node.asMethodDeclaration();
            this.key = fileName + "@" + methodDeclaration.getNameAsString();

            int line = methodDeclaration.getBegin().isPresent() ? methodDeclaration.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = methodDeclaration.getMetaModel().getTypeName();
            trace.append(type).append(' ');

            negSeq.put(tempKey, trace.toString());

            Optional<BlockStmt> body = methodDeclaration.getBody();
            if (body.isPresent()) {
                NodeList<Statement> statements = body.get().getStatements();
                for (Statement statement : statements) {
                    tempKey = dfs(statement, tempKey);
                }

            }
        }
        System.out.println();

    }

    /**
     * 传入当前节点和父亲节点的key
     *
     * @param node   当前节点
     * @param preKey 父亲节点的key
     */
    private String dfs(Statement node, String preKey) {
        if (node.isExpressionStmt()) {
            // TODO: 对于Expression的不同类型，也要进行考虑
            // 比如int a = getA() 这需要构建两个节点
            ExpressionStmt exStmt = ((ExpressionStmt) node).asExpressionStmt();
            Expression expression = exStmt.getExpression();
            String code = expression.toString();

            if (LogUtil.isLogStatement(code)) {
                //  如果是日志语句 那就把父亲节点移到posmap里 然后跳过自己
                posSeq.put(preKey, negSeq.get(preKey));
                negSeq.remove(preKey);
                return preKey;
            }
            int line = expression.getBegin().isPresent() ? expression.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = expression.getMetaModel().getTypeName();
            trace.append(type).append(' ');

            negSeq.put(tempKey, trace.toString());
            return tempKey;
            // 接下来我们处理可能带block的节点
            // 首先我们处理循环语句 因为循环比较简单
            // ————————————————————————————————————————————————————————————————————————————————————————————
        } else if (node.isWhileStmt()) {
            WhileStmt whileStmt = ((WhileStmt) node).asWhileStmt();
            int line = whileStmt.getBegin().isPresent() ? whileStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = whileStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());

            // 这里需要考虑 因为有body了 所以要对body进行dfs
            if (!whileStmt.getBody().isBlockStmt()) {
                dfs(whileStmt.getBody(), tempKey);
            } else {
                NodeList<Statement> statements = whileStmt.getBody().asBlockStmt().getStatements();
                for (Statement statement : statements) {
                    tempKey = dfs(statement, tempKey);
                }
            }
            // 还没完！因为每个block后都需要加一个标注块结束的控制节点！
            tempKey = this.key + "@" + line + "@endWhile";
            type = "endWhile";
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());

            return tempKey;
        } else if (node.isForStmt()) {
            ForStmt forStmt = ((ForStmt) node).asForStmt();
            int line = forStmt.getBegin().isPresent() ? forStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = forStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());

            // 这里需要考虑 因为有body了 所以要对body进行dfs
            if (!forStmt.getBody().isBlockStmt()) {
                dfs(forStmt.getBody(), tempKey);
            } else {
                NodeList<Statement> statements = forStmt.getBody().asBlockStmt().getStatements();
                for (Statement statement : statements) {
                    tempKey = dfs(statement, tempKey);
                }
            }
            // 还没完！因为每个block后都需要加一个标注块结束的控制节点！
            tempKey = this.key + "@" + line + "@endFor";
            type = "endFor";
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());

            return tempKey;
        } else if (node.isForeachStmt()) {
            ForeachStmt foreachStmt = ((ForeachStmt) node).asForeachStmt();
            int line = foreachStmt.getBegin().isPresent() ? foreachStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = foreachStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());

            // 这里需要考虑 因为有body了 所以要对body进行dfs
            if (!foreachStmt.getBody().isBlockStmt()) {
                dfs(foreachStmt.getBody(), tempKey);
            } else {
                NodeList<Statement> statements = foreachStmt.getBody().asBlockStmt().getStatements();
                for (Statement statement : statements) {
                    tempKey = dfs(statement, tempKey);
                }
            }
            // 还没完！因为每个block后都需要加一个标注块结束的控制节点！
            tempKey = this.key + "@" + line + "@endForeach";
            type = "endForeach";
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());

            return tempKey;

        } else if (node.isDoStmt()) {
            DoStmt doStmt = ((DoStmt) node).asDoStmt();
            int doLine = doStmt.getBegin().isPresent() ? doStmt.getBegin().get().line : -1;
            int whileLine = doStmt.getCondition().getBegin().isPresent() ? doStmt.getCondition().getBegin().get().line : -1;

            String tempKey = this.key + "@" + doLine;
            String type = "do";
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());

            // 这里需要考虑 因为有body了 所以要对body进行dfs
            if (!doStmt.getBody().isBlockStmt()) {
                dfs(doStmt.getBody(), tempKey);
            } else {
                NodeList<Statement> statements = doStmt.getBody().asBlockStmt().getStatements();
                for (Statement statement : statements) {
                    tempKey = dfs(statement, tempKey);
                }
            }
            // 还没完！do最后要wile语句
            tempKey = this.key + "@" + whileLine;
            // TODO: 这里需要看一下是不是DoStmt
            type = doStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());
            return tempKey;

            // 接下来依旧是可能带block的节点
            // 不过这次是条件分支 比较难写
            // ————————————————————————————————————————————————————————————————————————————————————————————
        } else if (node.isIfStmt()) {
            IfStmt ifStmt = ((IfStmt) node).asIfStmt();
            // 对于if语句 我们只需要顺序遍历then块和else块就好
            int line = ifStmt.getBegin().isPresent() ? ifStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = ifStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());

            if (!ifStmt.getThenStmt().isBlockStmt()) {
                tempKey = dfs(ifStmt.getThenStmt(), tempKey);
            } else {
                BlockStmt thenBlockStmt = ifStmt.getThenStmt().asBlockStmt();
                NodeList<Statement> statements = thenBlockStmt.getStatements();
                for (Statement statement : statements)
                    tempKey = dfs(statement, tempKey);
            }

            if (ifStmt.getElseStmt().isPresent()) {
                // 如果存在else块 要先判断是elif 还是else
                if (ifStmt.getElseStmt().get().isIfStmt()) {
                    tempKey = dfs(ifStmt.getElseStmt().get().asIfStmt(), tempKey);
                } else {
                    // 不然就是else
                    // else节点好像没有 得自己建一个
                    line = ifStmt.getElseStmt().get().getBegin().isPresent() ? ifStmt.getElseStmt().get().getBegin().get().line : -1;
                    tempKey = this.key + "@" + line;
                    String elseType = "else";
                    trace.append(elseType).append(' ');
                    negSeq.put(tempKey, trace.toString());

                    if (!ifStmt.getElseStmt().get().isBlockStmt()) {
                        tempKey = dfs(ifStmt.getElseStmt().get(), tempKey);
                    } else {
                        BlockStmt elseBlockStmt = ifStmt.getElseStmt().get().asBlockStmt();
                        NodeList<Statement> statements = elseBlockStmt.getStatements();
                        for (Statement statement : statements) {
                            tempKey = dfs(statement, tempKey);
                        }
                    }
                    // 还没完 对于ifStmt 最后要加一个endif块
                    tempKey = this.key + "@" + line + "@endIf";
                    type = "endIf";
                    trace.append(type).append(' ');
                    negSeq.put(tempKey, trace.toString());

                    return tempKey;
                }
            } else {
                // 当前节点没有else了 我们再考虑endif
                // 有else就接着遍历else 没else我们就用endif封上
                // else结束也要封
                // 还没完 对于ifStmt 最后要加一个endif块
                tempKey = this.key + "@" + line + "@endIf";
                type = "endIf";
                trace.append(type).append(' ');
                negSeq.put(tempKey, trace.toString());
                return tempKey;
            }

        } else if (node.isSwitchStmt()) {
            SwitchStmt switchStmt = ((SwitchStmt) node).asSwitchStmt();
            int line = switchStmt.getBegin().isPresent() ? switchStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = switchStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());

            NodeList<SwitchEntryStmt> caseEntries = switchStmt.getEntries();
            if (!caseEntries.isEmpty()) {
                for (SwitchEntryStmt caseEntry : caseEntries) {
                    tempKey = dfs(caseEntry, tempKey);
                    NodeList<Statement> statements = caseEntry.getStatements();
                    for (Statement statement : statements)
                        tempKey = dfs(statement, tempKey);
                }
            }

            // 没完！ 要接一个endSwtich
            line = switchStmt.getEnd().get().line + 1;
            tempKey = this.key + "@" + line;
            type = "endSwtich";
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());
            return tempKey;

            // 接下来处理一些块结构
            // ————————————————————————————————————————————————————————————————————————————————————————————
        } else if (node.isSynchronizedStmt()) {
            SynchronizedStmt synchronizedStmt = ((SynchronizedStmt) node).asSynchronizedStmt();
            int line = synchronizedStmt.getBegin().isPresent() ? synchronizedStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = synchronizedStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());

            // 这里需要考虑 因为有body了 所以要对body进行dfs
            if (!synchronizedStmt.getBody().isBlockStmt()) {
                dfs(synchronizedStmt.getBody(), tempKey);
            } else {
                NodeList<Statement> statements = synchronizedStmt.getBody().asBlockStmt().getStatements();
                for (Statement statement : statements) {
                    tempKey = dfs(statement, tempKey);
                }
            }
            // 还没完！因为每个block后都需要加一个标注块结束的控制节点！
            line = synchronizedStmt.getEnd().isPresent() ? synchronizedStmt.getEnd().get().line + 1 : line;
            tempKey = this.key + "@" + line + "@endSynchronized";
            type = "endSynchronized";
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());

            return tempKey;
        } else if (node.isBlockStmt()) {
            BlockStmt blockStmt = ((BlockStmt) node).asBlockStmt();
            String tempKey = preKey;
            NodeList<Statement> statements = blockStmt.getStatements();
            for (Statement statement : statements) {
                tempKey = dfs(statement, tempKey);
            }
            return tempKey;
            // 接下来处理一些简单语句
            // ————————————————————————————————————————————————————————————————————————————————————————————
        } else if (node.isSwitchEntryStmt()) {
            // TODO

        } else if (node.isBreakStmt()) {
            BreakStmt breakStmt = ((BreakStmt) node).asBreakStmt();
            int line = breakStmt.getBegin().isPresent() ? breakStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = breakStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());
            return tempKey;
        } else if (node.isContinueStmt()) {
            ContinueStmt continueStmt = ((ContinueStmt) node).asContinueStmt();
            int line = continueStmt.getBegin().isPresent() ? continueStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = continueStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());
            return tempKey;
        } else if (node.isLabeledStmt()) {
            LabeledStmt labeledStmt = ((LabeledStmt) node).asLabeledStmt();
            int line = labeledStmt.getBegin().isPresent() ? labeledStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = labeledStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());
            return tempKey;
        } else if (node.isReturnStmt()) {
            ReturnStmt returnStmt = ((ReturnStmt) node).asReturnStmt();
            int line = returnStmt.getBegin().isPresent() ? returnStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = returnStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());
            return tempKey;
        } else if (node.isEmptyStmt()) {
            EmptyStmt emptyStmt = ((EmptyStmt) node).asEmptyStmt();
            int line = emptyStmt.getBegin().isPresent() ? emptyStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = emptyStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());
            return tempKey;
        } else if (node.isAssertStmt()) {
            AssertStmt assertStmt = ((AssertStmt) node).asAssertStmt();
            int line = assertStmt.getBegin().isPresent() ? assertStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = assertStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());
            return tempKey;
        } else if (node.isExplicitConstructorInvocationStmt()) {
            ExplicitConstructorInvocationStmt explicitConstructorInvocationStmt = ((ExplicitConstructorInvocationStmt) node).asExplicitConstructorInvocationStmt();
            int line = explicitConstructorInvocationStmt.getBegin().isPresent() ? explicitConstructorInvocationStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = explicitConstructorInvocationStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());
            return tempKey;
        } else if (node.isLocalClassDeclarationStmt()) {
            LocalClassDeclarationStmt localClassDeclarationStmt = ((LocalClassDeclarationStmt) node).asLocalClassDeclarationStmt();
            int line = localClassDeclarationStmt.getBegin().isPresent() ? localClassDeclarationStmt.getBegin().get().line : -1;
            String tempKey = this.key + "@" + line;
            String type = localClassDeclarationStmt.getMetaModel().getTypeName();
            trace.append(type).append(' ');
            negSeq.put(tempKey, trace.toString());
            return tempKey;

            // 最后我们来处理trycatch
        } else if (node.isTryStmt()) {
            
        }


        return null;
    }


}
