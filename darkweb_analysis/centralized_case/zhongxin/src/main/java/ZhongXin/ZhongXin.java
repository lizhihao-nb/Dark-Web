package ZhongXin;

import org.janusgraph.core.JanusGraph;
import org.janusgraph.core.JanusGraphFactory;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.process.traversal.P;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__;

import java.util.*;
import java.io.FileWriter;
import java.io.IOException;

public class ZhongXin {

    private static final String CATEGORY = "Darknet Forums & Directories";

    public static void main(String[] args) {
        String configPath = "/public/home/blockchain_2/slave1/darkanalysis/janusgraph-hbase-solr4.properties";
        
        JanusGraph graph = loadGraph(configPath);
        if (graph == null) {
            System.err.println("❌ 无法加载图数据库");
            return;
        }
        GraphTraversalSource g = graph.traversal();

        try {
            System.out.println("🔍 正在提取 Darknet Forums & Directories 的多个连通分量 (100–200 节点)...");
            extractMultipleComponents(g, CATEGORY, 5, 200);
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
            int compSize = currentComp.size();
            // ✅ 严格限定 100 <= size <= 200
            if (compSize >= 100 && compSize <= maxSize) {
                savedCount++;
                saveComponentAsJson(g, currentComp, "component_" + savedCount + ".json");
                System.out.println("  已保存 component_" + savedCount + ".json (" + compSize + " 节点)");
                if (savedCount >= numComponents) break;
            }
        }

        if (savedCount == 0) {
            throw new RuntimeException("未找到任何 100–200 节点范围内的连通分量");
        }
    }

    private static void saveComponentAsJson(
        GraphTraversalSource g,
        Set<String> nodeIds,
        String filename
    ) throws IOException {
        List<Map<String, String>> edges = new ArrayList<>();
        for (String src : nodeIds) {
            List<String> targets = new ArrayList<>();
            for (Object tgt : g.V().has("bulkLoader.vertex.id", src)
                    .out()
                    .has("category", CATEGORY)
                    .has("bulkLoader.vertex.id", P.within(nodeIds))
                    .values("bulkLoader.vertex.id")
                    .toList()) {
                targets.add(tgt.toString());
            }
            for (String tgt : targets) {
                Map<String, String> edge = new HashMap<>();
                edge.put("source", src);
                edge.put("target", tgt);
                edges.add(edge);
            }
        }

        StringBuilder json = new StringBuilder();
        json.append("{\n  \"nodes\": [\n");
        boolean firstNode = true;
        for (String id : nodeIds) {
            if (!firstNode) json.append(",\n");
            json.append("    {\"id\": \"").append(escapeJson(id)).append("\", \"label\": \"").append(CATEGORY).append("\"}");
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