package CategoryLink;

import org.janusgraph.core.JanusGraph;
import org.janusgraph.core.JanusGraphFactory;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.structure.Vertex;
import org.apache.tinkerpop.gremlin.structure.Edge;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__;

import java.util.*;
import java.io.FileWriter;
import java.io.IOException;

public class CategoryLink {
    // 预定义的11个类别（按你的顺序）
    private static final List<String> PREDEFINED_CATEGORIES = Arrays.asList(
        "Illicit Marketplaces",
        "Darknet Forums & Directories", 
        "Financial Fraud & Money Laundering",
        "Hacking & Malware",
        "Illicit Drug Trade",
        "Firearms Trafficking", 
        "Child Sexual Exploitation",
        "Anonymity & OPSEC Guidance",
        "Illicit Hosting Infrastructure",
        "Digital Archiving & Preservation",
        "Law & Political Reporting"
    );
    
    public static void main(String[] args) {
        String janusGraphConfig = "/public/home/blockchain_2/slave1/darkanalysis/janusgraph-hbase-solr4.properties";
        
        // 连接图数据库
        JanusGraph graph = JanusGraphFactory.open(janusGraphConfig);
        GraphTraversalSource g = graph.traversal();
        
        try {
            System.out.println("开始进行类型间连接统计分析...");
            
            List<String> categories = PREDEFINED_CATEGORIES;
            System.out.println("使用的类别: " + categories);
            System.out.println("类别总数: " + categories.size());
            
            // 创建类别索引映射
            Map<String, Integer> categoryIndex = new HashMap<>();
            for (int i = 0; i < categories.size(); i++) {
                categoryIndex.put(categories.get(i), i);
            }
            
            int size = categories.size();
            long[][] connectionMatrix = new long[size][size];
            long[] outDegree = new long[size];
            
            // 获取所有连接两个有 category 节点的边
            // System.out.println("正在统计连接关系...");
            // List<Edge> edges = g.E()
            //     .where(__.outV().has("category"))
            //     .where(__.inV().has("category"))
            //     .toList();
            List<Edge> edges = g.E().toList();
            
            System.out.println("找到 " + edges.size() + " 条有效连接边");
            
            int validEdges = 0;
            for (Edge edge : edges) {
                Vertex outVertex = edge.outVertex();
                Vertex inVertex = edge.inVertex();
                
                String sourceCategory = getCategoryValue(outVertex);
                String targetCategory = getCategoryValue(inVertex);
                
                if (sourceCategory != null && targetCategory != null &&
                    categoryIndex.containsKey(sourceCategory) && 
                    categoryIndex.containsKey(targetCategory)) {
                    
                    int sourceIdx = categoryIndex.get(sourceCategory);
                    int targetIdx = categoryIndex.get(targetCategory);
                    
                    connectionMatrix[sourceIdx][targetIdx]++;
                    outDegree[sourceIdx]++;
                    validEdges++;
                }
            }
            
            System.out.println("有效连接边（在预定义类别中）: " + validEdges);
            
            // 计算连接偏好度 (标准化)
            double[][] preferenceMatrix = new double[size][size];
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    if (outDegree[i] > 0) {
                        preferenceMatrix[i][j] = (double) connectionMatrix[i][j] / outDegree[i];
                    } else {
                        preferenceMatrix[i][j] = 0.0;
                    }
                }
            }
            
            // 找出最强的连接模式
            System.out.println("\n=== 最强连接模式 Top 15 ===");
            findTopConnections(connectionMatrix, categories, 15);
            
            // 分析自连接比例
            analyzeSelfConnections(connectionMatrix, outDegree, categories);
            
            // 导出为矩阵形式的 CSV 文件
            exportMatrixToCSV(connectionMatrix, preferenceMatrix, categories);
            
            System.out.println("\n✅ 分析完成！");
            
        } catch (Exception e) {
            System.err.println("❌ 分析过程中出现错误:");
            e.printStackTrace();
        } finally {
            graph.close();
        }
    }
    
    // 安全获取 category 值（处理多值属性）
    private static String getCategoryValue(Vertex vertex) {
        try {
            Object value = vertex.value("category");
            if (value == null) {
                return null;
            }
            if (value instanceof List) {
                List<?> list = (List<?>) value;
                if (!list.isEmpty()) {
                    return list.get(0).toString();
                }
                return null;
            }
            return value.toString();
        } catch (Exception e) {
            return null;
        }
    }
    
    // 找出最强的连接模式
    private static void findTopConnections(long[][] matrix, List<String> categories, int topN) {
        List<ConnectionInfo> connections = new ArrayList<>();
        for (int i = 0; i < categories.size(); i++) {
            for (int j = 0; j < categories.size(); j++) {
                if (matrix[i][j] > 0) {
                    connections.add(new ConnectionInfo(
                        categories.get(i), 
                        categories.get(j), 
                        matrix[i][j]
                    ));
                }
            }
        }
        connections.sort((a, b) -> Long.compare(b.count, a.count));
        int count = Math.min(topN, connections.size());
        for (int i = 0; i < count; i++) {
            ConnectionInfo conn = connections.get(i);
            System.out.printf("%2d. %s → %s: %d 条连接\n", 
                i + 1, conn.source, conn.target, conn.count);
        }
    }
    
    // 分析自连接比例
    private static void analyzeSelfConnections(long[][] matrix, long[] outDegree, List<String> categories) {
        System.out.println("\n=== 自连接分析 ===");
        long totalSelfConnections = 0;
        long totalConnections = 0;
        for (int i = 0; i < categories.size(); i++) {
            totalSelfConnections += matrix[i][i];
            totalConnections += outDegree[i];
        }
        System.out.printf("总自连接数: %d\n", totalSelfConnections);
        System.out.printf("总连接数: %d\n", totalConnections);
        System.out.printf("自连接比例: %.2f%%\n", 
            totalConnections > 0 ? (double) totalSelfConnections / totalConnections * 100 : 0);
        
        System.out.println("\n各类型自连接比例:");
        for (int i = 0; i < categories.size(); i++) {
            if (outDegree[i] > 0) {
                double selfRatio = (double) matrix[i][i] / outDegree[i];
                System.out.printf("%s: %.2f%% (%d/%d)\n", 
                    categories.get(i), selfRatio * 100, matrix[i][i], outDegree[i]);
            } else {
                System.out.printf("%s: 0.00%% (0/0)\n", categories.get(i));
            }
        }
    }
    
    // 导出为矩阵形式的 CSV（带行列标签）
    private static void exportMatrixToCSV(long[][] connectionMatrix, double[][] preferenceMatrix, List<String> categories) {
        String[] filenames = {"connection_matrix.csv", "preference_matrix.csv"};
        for (int m = 0; m < 2; m++) {
            try (FileWriter writer = new FileWriter(filenames[m])) {
                // 写入表头
                writer.write("\"Source\"");
                for (String target : categories) {
                    writer.write(",\"" + escapeCsv(target) + "\"");
                }
                writer.write("\n");

                // 写入数据行
                for (int i = 0; i < categories.size(); i++) {
                    writer.write("\"" + escapeCsv(categories.get(i)) + "\"");
                    for (int j = 0; j < categories.size(); j++) {
                        if (m == 0) {
                            writer.write("," + connectionMatrix[i][j]);
                        } else {
                            writer.write(String.format(",%.6f", preferenceMatrix[i][j]));
                        }
                    }
                    writer.write("\n");
                }
            } catch (IOException e) {
                System.err.println("❌ 无法写入矩阵 CSV 文件 (" + filenames[m] + "): " + e.getMessage());
            }
        }
        System.out.println("✅ 矩阵已导出:");
        System.out.println("   - 原始连接矩阵: connection_matrix.csv");
        System.out.println("   - 偏好度矩阵: preference_matrix.csv");
    }

    // CSV 字段转义（处理逗号、引号、换行符）
    private static String escapeCsv(String value) {
        if (value == null) return "";
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }
    
    // 连接信息内部类
    private static class ConnectionInfo {
        String source;
        String target;
        long count;
        ConnectionInfo(String source, String target, long count) {
            this.source = source;
            this.target = target;
            this.count = count;
        }
    }
}