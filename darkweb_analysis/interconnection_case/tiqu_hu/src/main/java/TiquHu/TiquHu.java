package TiquHu;

import org.janusgraph.core.JanusGraph;
import org.janusgraph.core.JanusGraphFactory;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.process.traversal.P; // ✅ 关键导入
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class TiquHu {

    private static final String MARKET = "Illicit Marketplaces";
    private static final String FINANCE = "Financial Fraud & Money Laundering";

    public static void main(String[] args) {
        String configPath = "/public/home/blockchain_2/slave1/darkanalysis/janusgraph-hbase-solr4.properties";
        JanusGraph graph = null;

        try {
            graph = JanusGraphFactory.open(configPath);
            GraphTraversalSource g = graph.traversal();

            // Step 1: 获取所有 Market 和 Finance 节点 ID
            System.out.println("正在获取 Market 和 Finance 节点...");
            List<Object> marketIdObjects = g.V().has("category", MARKET).values("bulkLoader.vertex.id").toList();
            List<Object> financeIdObjects = g.V().has("category", FINANCE).values("bulkLoader.vertex.id").toList();

            List<String> marketIds = new ArrayList<>();
            for (Object id : marketIdObjects) {
                marketIds.add(id.toString());
            }
            List<String> financeIds = new ArrayList<>();
            for (Object id : financeIdObjects) {
                financeIds.add(id.toString());
            }

            Set<String> marketSet = new HashSet<>(marketIds);
            Set<String> financeSet = new HashSet<>(financeIds);
            Set<String> allTargetIds = new HashSet<>(marketSet);
            allTargetIds.addAll(financeSet);

            System.out.println("找到 Market 节点: " + marketIds.size());
            System.out.println("找到 Finance 节点: " + financeIds.size());

            // Step 2: 构建节点标签映射
            Map<String, String> nodeLabels = new HashMap<>();
            for (String id : marketIds) nodeLabels.put(id, "marketNode");
            for (String id : financeIds) nodeLabels.put(id, "financeNode");

            // Step 3: 收集边
            List<Map<String, String>> edgesList = new ArrayList<>();
            Set<String> processedEdges = new HashSet<>();

            collectEdgesFromIds(g, marketIds, allTargetIds, edgesList, processedEdges);
            collectEdgesFromIds(g, financeIds, allTargetIds, edgesList, processedEdges);

            // 构建 nodes 列表
            List<Map<String, String>> nodesList = new ArrayList<>();
            for (String id : allTargetIds) {
                Map<String, String> node = new HashMap<>();
                node.put("id", id);
                node.put("label", nodeLabels.get(id));
                nodesList.add(node);
            }

            exportToJson(nodesList, edgesList);
            System.out.println("✅ 成功生成 data.js: " + allTargetIds.size() + " 节点, " + edgesList.size() + " 边");

        } catch (Exception e) {
            System.err.println("❌ 出错:");
            e.printStackTrace();
        } finally {
            if (graph != null) {
                try {
                    graph.close();
                } catch (Exception e) {
                    System.err.println("关闭图失败: " + e.getMessage());
                }
            }
        }
    }

    private static void collectEdgesFromIds(
            GraphTraversalSource g,
            List<String> sourceIds,
            Set<String> targetIdSet,
            List<Map<String, String>> edgesList,
            Set<String> processedEdges) {

        // ✅ 直接使用 P.within(targetIdSet)，无需转数组！
        for (String srcId : sourceIds) {
            g.V().has("bulkLoader.vertex.id", srcId)
                .outE().inV()
                .has("bulkLoader.vertex.id", P.within(targetIdSet)) // ✅ 正确用法
                .project("tgt_id")
                    .by(__.values("bulkLoader.vertex.id"))
                .forEachRemaining(map -> {
                    String tgtId = map.get("tgt_id").toString();
                    String edgeKey = srcId + "->" + tgtId;
                    if (!processedEdges.contains(edgeKey)) {
                        processedEdges.add(edgeKey);
                        Map<String, String> edge = new HashMap<>();
                        edge.put("source", srcId);
                        edge.put("target", tgtId);
                        edgesList.add(edge);
                    }
                });
        }
    }

    private static void exportToJson(List<Map<String, String>> nodes, List<Map<String, String>> edges) {
        try (FileWriter writer = new FileWriter("data.js")) {
            StringBuilder json = new StringBuilder();
            json.append("const data = {\n");
            json.append("  \"nodes\": [\n");
            for (int i = 0; i < nodes.size(); i++) {
                Map<String, String> node = nodes.get(i);
                json.append("    {\"id\": \"").append(escapeJson(node.get("id")))
                    .append("\", \"label\": \"").append(escapeJson(node.get("label"))).append("\"}");
                if (i < nodes.size() - 1) json.append(",");
                json.append("\n");
            }
            json.append("  ],\n");
            json.append("  \"edges\": [\n");
            for (int i = 0; i < edges.size(); i++) {
                Map<String, String> edge = edges.get(i);
                json.append("    {\"source\": \"").append(escapeJson(edge.get("source")))
                    .append("\", \"target\": \"").append(escapeJson(edge.get("target"))).append("\"}");
                if (i < edges.size() - 1) json.append(",");
                json.append("\n");
            }
            json.append("  ]\n};\n");
            writer.write(json.toString());
        } catch (IOException e) {
            throw new RuntimeException("写入失败", e);
        }
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
    }
}