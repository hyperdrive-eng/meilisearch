// Let's modify the test to be more explicit about reproducing the bug
#[actix_rt::test]
async fn test_hybrid_search_distinct_bug_reproduction() {
    use std::collections::HashSet;
    
    let server = Server::new().await;
    let index = server.index("test");

    // Set up embedder
    let (response, code) = index
        .update_settings(json!({ 
            "embedders": {
                "default": {
                    "source": "userProvided",
                    "dimensions": 2
                }
            }
        }))
        .await;
    assert_eq!(202, code);
    index.wait_task(response.uid()).await.succeeded();

    // Create documents designed to trigger the bug:
    // - Same product_id but very different content that will rank differently
    // - Ensure one ranks high in keyword search, other in vector search
    let test_documents = json!([
        {
            "id": 1,
            "title": "NIKE SPORTS JACKET LEATHER RED",  // High keyword match
            "brand": "Nike", 
            "product_id": "SAME_PRODUCT",
            "category": "sports",
            "_vectors": {"default": [0.1, 0.1]}  // Low vector similarity
        },
        {
            "id": 2,
            "title": "Blue casual wear item", // Low keyword match
            "brand": "Nike",
            "product_id": "SAME_PRODUCT", // SAME product_id!
            "category": "fashion",
            "_vectors": {"default": [0.9, 0.9]} // High vector similarity to search vector
        },
        {
            "id": 3,
            "title": "Different product entirely",
            "brand": "Adidas",
            "product_id": "DIFFERENT_PRODUCT",
            "category": "other",
            "_vectors": {"default": [0.5, 0.5]}
        }
    ]);

    let (response, code) = index.add_documents(test_documents, Some("id")).await;
    assert_eq!(202, code);
    index.wait_task(response.uid()).await.succeeded();

    // Set distinct on product_id
    let (task, _) = index.update_distinct_attribute(json!("product_id")).await;
    index.wait_task(task.uid()).await.succeeded();

    // Search that should favor document 1 for keywords and document 2 for vectors
    let (response, code) = index
        .search_post(json!({
            "q": "NIKE SPORTS JACKET LEATHER RED",  // Should strongly match doc 1
            "vector": [0.9, 0.9],  // Should strongly match doc 2
            "hybrid": {
                "embedder": "default", 
                "semanticRatio": 0.5  // Equal weight to both searches
            },
            "limit": 10
        }))
        .await;

    assert_eq!(200, code);
    let hits = response["hits"].as_array().unwrap();
    
    println!("Total hits: {}", hits.len());
    for (i, hit) in hits.iter().enumerate() {
        println!("Hit {}: id={}, product_id={}, title={}", 
                i, 
                hit["id"], 
                hit["product_id"], 
                hit["title"]);
    }
    
    // Count occurrences of each product_id
    let mut product_id_counts = std::collections::HashMap::new();
    for hit in hits {
        if let Some(product_id) = hit["product_id"].as_str() {
            *product_id_counts.entry(product_id).or_insert(0) += 1;
        }
    }
    
    println!("Product ID counts: {:?}", product_id_counts);
    
    // Check if SAME_PRODUCT appears more than once - this would be the bug
    if let Some(&count) = product_id_counts.get("SAME_PRODUCT") {
        if count > 1 {
            panic!("BUG REPRODUCED! product_id 'SAME_PRODUCT' appears {} times in hybrid search results, violating distinct constraint", count);
        }
    }
    
    println!("Distinct filtering appears to be working correctly");
}
