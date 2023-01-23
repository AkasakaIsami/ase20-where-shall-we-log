import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.visitor.VoidVisitor;
import config.Config;
import visitor.MethodVisitor;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import static config.Config.targetDir4Neg;
import static config.Config.targetDir4Pos;

/**
 * <p>You can check the problem detail on <a href="">Leetcode</a>.</p>
 *
 * @author Akasaka Isami
 * @since 2023/1/17 9:05 PM
 */
public class Entry {

    public static void main(String[] args) throws IOException {

        String dataSrc = Config.rootDir;
        File srcDir = new File(dataSrc);
        if (!srcDir.isDirectory() || !srcDir.exists()) {
            System.out.println("找不到源码！检查" + Config.rootDir + "是不是有源码？");
            return;
        }

        File tar1 = new File(targetDir4Neg);
        File tar2 = new File(targetDir4Pos);

        if (!new File(targetDir4Neg).exists())
            tar1.mkdirs();
        if (!new File(targetDir4Pos).exists())
            tar2.mkdirs();

        FileWriter negFw = new FileWriter(targetDir4Neg + "data.txt");
        FileWriter posFw = new FileWriter(targetDir4Pos + "data.txt");

        // 如果有 遍历所有java文件
        for (File file : Objects.requireNonNull(srcDir.listFiles())) {
            String fileName = file.getName();


            try {
                CompilationUnit cu = JavaParser.parse(file);

                VoidVisitor<String> methodVisitor = new MethodVisitor();
                methodVisitor.visit(cu, fileName);

            } catch (ParseProblemException e1) {
                System.out.println(fileName + "解析出错，直接跳过");
            } catch (FileNotFoundException e2) {
                System.out.println(fileName + "文件没找到，直接跳过");
            }
        }

        save_data(MethodVisitor.negSeq, negFw);
        save_data(MethodVisitor.posSeq, posFw);

        negFw.close();
        posFw.close();
    }

    private static void save_data(Map<String, String> seqs, FileWriter fw) throws IOException {
        for (Map.Entry<String, String> entry : seqs.entrySet()) {
            String key = entry.getKey();
            String seq = entry.getValue();
            fw.write(key + ' ' + seq + "\n");
        }
    }


}
