package QuZhong;

import org.janusgraph.core.JanusGraph;
import org.janusgraph.core.JanusGraphFactory;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.structure.Vertex;
import org.apache.tinkerpop.gremlin.process.traversal.P;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__;

import java.util.*;
import java.io.FileWriter;
import java.io.IOException;

public class QuZhong {

    private static final String MARKET = "Illicit Drug Trade";

    public static void main(String[] args) {
        String configPath = "/public/home/blockchain_2/slave1/darkanalysis/janusgraph-hbase-solr4.properties";
        
        JanusGraph graph = loadGraph(configPath);
        if (graph == null) {
            System.err.println("❌ 无法加载图数据库");
            return;
        }
        GraphTraversalSource g = graph.traversal();

        try {
            System.out.println("🔍 正在提取 Illicit Drug Trade 的多个连通分量...");
            extractMultipleComponents(g, MARKET, 20, 100);
            System.out.println("✅ 所有分量已保存为 component_X.json");
        } catch (Exception e) {
            System.err.println("❌ 提取失败:");
            e.printStackTrace();
        } finally {
            graph.close();
        }
    }

    private static JanusGraph loadGraph(String configPath) {
        try {
            return JanusGraphFactory.open(configPath);
        } catch (Exception e) {
            System.err.println("⚠️ 图加载失败: " + e.getMessage());
            return null;
        }
    }

    private static void extractMultipleComponents(
        GraphTraversalSource g,
        String category,
        int numComponents,
        int maxSize
    ) throws IOException {
        Set<String> allVisited = new HashSet<>();
        // 修复 1: 显式转换 Object -> String
        List<String> allIds = new ArrayList<>();
        for (Object id : g.V().has("category", category).values("bulkLoader.vertex.id").toList()) {
            allIds.add(id.toString());
        }
        System.out.println("  总节点数: " + allIds.size());

        int savedCount = 0;
        for (String startId : allIds) {
            if (allVisited.contains(startId)) continue;
            if (savedCount >= numComponents) break;

            Set<String> currentComp = new LinkedHashSet<>();
            Queue<String> queue = new LinkedList<>();
            queue.add(startId);
            currentComp.add(startId);

            while (!queue.isEmpty()) {
                String curId = queue.poll();
                // 修复 1: 显式转换
                List<String> neighbors = new ArrayList<>();
                for (Object nid : g.V().has("bulkLoader.vertex.id", curId)
                        .both()
                        .has("category", category)
                        .values("bulkLoader.vertex.id")
                        .toList()) {
                    neighbors.add(nid.toString());
                }
                for (String nid : neighbors) {
                    if (!currentComp.contains(nid)) {
                        currentComp.add(nid);
                        queue.add(nid);
                    }
                }
            }

            allVisited.addAll(currentComp);
            if (currentComp.size() >= 30 && currentComp.size() <= maxSize) {
                savedCount++;
                saveComponentAsJson(g, currentComp, "component_" + savedCount + ".json");
                System.out.println("  已保存 component_" + savedCount + ".json (" + currentComp.size() + " 节点)");
                if (savedCount >= numComponents) break;
            }
        }

        if (savedCount == 0) {
            throw new RuntimeException("未找到任何符合条件的连通分量");
        }
    }

    private static void saveComponentAsJson(
        GraphTraversalSource g,
        Set<String> nodeIds,
        String filename
    ) throws IOException {
        // 收集边
        List<Map<String, String>> edges = new ArrayList<>();
        for (String src : nodeIds) {
            // 修复 1: 显式转换
            List<String> targets = new ArrayList<>();
            for (Object tgt : g.V().has("bulkLoader.vertex.id", src)
                    .out()
                    .has("category", MARKET)
                    .has("bulkLoader.vertex.id", P.within(nodeIds))
                    .values("bulkLoader.vertex.id")
                    .toList()) {
                targets.add(tgt.toString());
            }
            for (String tgt : targets) {
                // 修复 2: Java 8 兼容的 Map 创建方式
                Map<String, String> edge = new HashMap<>();
                edge.put("source", src);
                edge.put("target", tgt);
                edges.add(edge);
            }
        }

        // 写入 JSON
        StringBuilder json = new StringBuilder();
        json.append("{\n  \"nodes\": [\n");
        boolean firstNode = true;
        for (String id : nodeIds) {
            if (!firstNode) json.append(",\n");
            json.append("    {\"id\": \"").append(escapeJson(id)).append("\", \"label\": \"Illicit Drug Trade\"}");
            firstNode = false;
        }
        json.append("\n  ],\n  \"edges\": [\n");
        boolean firstEdge = true;
        for (Map<String, String> e : edges) {
            if (!firstEdge) json.append(",\n");
            json.append("    {\"source\": \"").append(escapeJson(e.get("source")))
                .append("\", \"target\": \"").append(escapeJson(e.get("target"))).append("\"}");
            firstEdge = false;
        }
        json.append("\n  ]\n}");

        try (FileWriter writer = new FileWriter(filename)) {
            writer.write(json.toString());
        }
    }

    private static String escapeJson(String str) {
        return str.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}