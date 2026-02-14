package YiZhi;

import org.janusgraph.core.JanusGraph;
import org.janusgraph.core.JanusGraphFactory;
import org.apache.tinkerpop.gremlin.process.traversal.P;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.structure.Edge;
import org.apache.tinkerpop.gremlin.structure.Vertex;
import org.apache.tinkerpop.gremlin.process.traversal.P;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class YiZhi {

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

        JanusGraph graph = JanusGraphFactory.open(janusGraphConfig);
        GraphTraversalSource g = graph.traversal();

        try {
            System.out.println("🔍 开始分析各暗网类别的内部连通分量结构（仅 size ≥ 4，使用内存邻接表优化）...");
            analyzeDegreeHeterogeneityPerCategoryToFile(g, PREDEFINED_CATEGORIES);
            System.out.println("\n✅ 分析完成！每个类别结果已保存为独立 CSV 文件。");

        } catch (Exception e) {
            System.err.println("❌ 分析过程中发生错误:");
            e.printStackTrace();
        } finally {
            graph.close();
        }
    }

    /**
     * 高性能版本：预加载同类子图到内存邻接表，仅分析 size ≥ 4 的连通分量
     */
    private static void analyzeDegreeHeterogeneityPerCategoryToFile(GraphTraversalSource g, List<String> categories) {
        final int MIN_COMPONENT_SIZE = 4;

        for (String category : categories) {
            System.out.println("🔍 处理类别: " + category);

            // Step 1: 获取所有该类别的节点
            List<Vertex> allNodesList = g.V().has("category", category).toList();
            if (allNodesList.isEmpty()) {
                System.out.println("  → 无节点，跳过");
                continue;
            }

            Set<Vertex> nodeSet = new HashSet<>(allNodesList);
            String safeCategory = category.replaceAll("[/\\\\?%*:|\"<>\n]", "_")
                                          .replaceAll("\\s+", "_")
                                          .replaceAll("_+", "_");
            String filename = safeCategory + ".csv";

            try (FileWriter writer = new FileWriter(filename)) {
                writer.write("ComponentID,NodeCount,MeanDegree,StdDev,Heterogeneity\n");

                // Step 2: 构建邻接表（无向图）
                Map<Vertex, List<Vertex>> adj = new HashMap<>();
                for (Vertex v : nodeSet) {
                    adj.put(v, new ArrayList<>());
                }

                // 获取所有两端都是当前类别的边（无向视角）
                List<Edge> internalEdges = g.E()
                    .where(__.outV().has("category", category))
                    .where(__.inV().has("category", category))
                    .toList();

                for (Edge e : internalEdges) {
                    Vertex out = e.outVertex();
                    Vertex in = e.inVertex();
                    // 无向连接：双向添加
                    if (nodeSet.contains(out) && nodeSet.contains(in)) {
                        adj.get(out).add(in);
                        if (!out.equals(in)) { // 避免自环重复加
                            adj.get(in).add(out);
                        }
                    }
                }

                // Step 3: 内存 BFS 找连通分量
                Set<Vertex> visited = new HashSet<>();
                int compId = 0;
                int skippedSmall = 0;

                for (Vertex start : nodeSet) {
                    if (visited.contains(start)) continue;

                    List<Vertex> component = new ArrayList<>();
                    Queue<Vertex> queue = new LinkedList<>();
                    queue.add(start);
                    visited.add(start);

                    while (!queue.isEmpty()) {
                        Vertex current = queue.poll();
                        component.add(current);

                        for (Vertex neighbor : adj.get(current)) {
                            if (!visited.contains(neighbor)) {
                                visited.add(neighbor);
                                queue.add(neighbor);
                            }
                        }
                    }

                    // 仅保留 size >= 4 的分量
                    if (component.size() < MIN_COMPONENT_SIZE) {
                        skippedSmall++;
                        continue;
                    }

                    // Step 4: 计算度异质性 H（直接用邻接表度数）
                    List<Integer> degrees = new ArrayList<>();
                    for (Vertex v : component) {
                        degrees.add(adj.get(v).size());
                    }

                    double mean = degrees.stream().mapToInt(Integer::intValue).average().orElse(0.0);
                    double stdDev = 0.0;
                    if (mean > 0) {
                        double sumSq = degrees.stream()
                            .mapToDouble(d -> Math.pow(d - mean, 2))
                            .sum();
                        stdDev = Math.sqrt(sumSq / component.size()); // 总体标准差
                    }
                    double heterogeneity = (mean > 0) ? stdDev / mean : 0.0;

                    // 写入 CSV
                    writer.write(String.format(
                        "%d,%d,%.6f,%.6f,%.6f\n",
                        compId,
                        component.size(),
                        mean,
                        stdDev,
                        heterogeneity
                    ));

                    compId++;
                }

                System.out.printf("  → 有效分量 (≥%d): %d, 小分量 (<%d): %d\n",
                    MIN_COMPONENT_SIZE, compId, MIN_COMPONENT_SIZE, skippedSmall);

            } catch (IOException e) {
                System.err.println("❌ 无法写入文件 " + filename + ": " + e.getMessage());
            }
        }
    }
}