package util;

import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author Akasaka Isami
 * @description 日志语句相关的工具类
 * <a href="https://regexr-cn.com/">参考这个网址</>
 */
public class LogUtil {

    private static final String regexGLOBAL = ".*?(log|trace|(system\\.out)|(system\\.err)).*?(.*?)";

    private static final Set<String> level_prix = new HashSet<String>() {{
        add("log");
        add("logger");
        add("logging");
        add("getlogger");
        add("getlog");
    }};

    private static final Set<String> levels = new HashSet<String>() {{
        add("trace");
        add("debug");
        add("info");
        add("warn");
        add("error");
        add("fatal");
    }};

    public static boolean isLogStatement(String statement) {
        String myRegex = "(log|logger|logging|getlogger|getlog)\\.(trace|debug|info|warn|error|fatal)\\(.*?\\)";
        Pattern p = Pattern.compile(myRegex, Pattern.CASE_INSENSITIVE);
        Matcher m = p.matcher(statement);
        return m.find();
    }

}
