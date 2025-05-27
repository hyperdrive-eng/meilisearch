#[actix_rt::test]
async fn test_hybrid_distinct_bug_extreme() {
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

    // Documents with IDENTICAL distinct values that should appear in BOTH searches
    let test_documents = json!([
        {
            "id": 1,
            "title": "red shoes nike",  // Should match keyword search well
            "product_id": "DUPLICATE_PRODUCT_ID",
            "_vectors": {"default": [0.0, 0.0]}  // Low similarity to search vector
        },
        {
            "id": 2,
            "title": "green shirt adidas",  // Different content
            "product_id": "DUPLICATE_PRODUCT_ID",  // SAME product_id!
            "_vectors": {"default": [1.0, 1.0]}   // High similarity to search vector
        }
    ]);

    let (response, code) = index.add_documents(test_documents, Some("id")).await;
    assert_eq!(202, code);
    index.wait_task(response.uid()).await.succeeded();

    // Set distinct on product_id
    let (task, _) = index.update_distinct_attribute(json!("product_id")).await;
    index.wait_task(task.uid()).await.succeeded();

    // Hybrid search - should trigger both vector and keyword search
    let (response, code) = index
        .search_post(json!({
            "q": "red shoes nike",      // Should rank doc 1 high in keyword search
            "vector": [1.0, 1.0],       // Should rank doc 2 high in vector search  
            "hybrid": {
                "embedder": "default", 
                "semanticRatio": 0.5
            },
            "limit": 10
        }))
        .await;

    assert_eq!(200, code);
    let hits = response["hits"].as_array().unwrap();
    
    println!("SEARCH RESULTS:");
    for (i, hit) in hits.iter().enumerate() {
        println!("  Hit {}: id={}, product_id={}", i, hit["id"], hit["product_id"]);
    }
    
    // Check for duplicate product_ids - THIS IS THE BUG
    let product_ids: Vec<&str> = hits.iter()
        .filter_map(|hit| hit["product_id"].as_str())
        .collect();
    
    let unique_count = product_ids.iter().collect::<std::collections::HashSet<_>>().len();
    
    if product_ids.len() > unique_count {
        panic!("BUG REPRODUCED! Found {} total results but only {} unique product_ids. \
               This means distinct filtering failed in hybrid search! Results: {:?}", 
               product_ids.len(), unique_count, product_ids);
    }
    
    // If only one result, distinct filtering worked (but we hoped to reproduce the bug)
    if hits.len() == 1 {
        println!("Only one result returned - distinct filtering working or different issue");
    } else {
        println!("Multiple results with unique product_ids - distinct filtering appears to work");
    }
}
