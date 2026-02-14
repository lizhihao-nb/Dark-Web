def defineGratefulDeadSchema(janusGraph) {
    m = janusGraph.openManagement()

    dark = m.makeVertexLabel("dark").make()
    surface = m.makeVertexLabel("surface").make()
    flowby = m.makeEdgeLabel("flow").make()

    blid  = m.makePropertyKey("bulkLoader.vertex.id").dataType(String.class).make()
    blid_ed  = m.makePropertyKey("bulkLoader.edge.id").dataType(String.class).make()
    value_ = m.makePropertyKey("value").dataType(String.class).make()
    level = m.makePropertyKey("level").dataType(String.class).make()
    edge_kind = m.makePropertyKey("edge_kind").dataType(String.class).make()
    site = m.makePropertyKey("site").dataType(String.class).make()

    // 新增：category 属性
    category = m.makePropertyKey("category").dataType(String.class).make()

    // 为 vertex 创建索引（包括新增的 category）
    m.buildIndex("byBulkLoaderVertexId", Vertex.class).addKey(blid).buildCompositeIndex()
    m.buildIndex("value_index", Vertex.class).addKey(value_).buildCompositeIndex()
    m.buildIndex("level_index", Vertex.class).addKey(level).buildCompositeIndex()
    m.buildIndex("site_index", Vertex.class).addKey(site).buildCompositeIndex()
    m.buildIndex("category_index", Vertex.class).addKey(category).buildCompositeIndex()  // 新增索引

    // Edge 索引
    m.buildIndex("edge_kind_index", Edge.class).addKey(edge_kind).buildCompositeIndex()
    
    m.commit()
}