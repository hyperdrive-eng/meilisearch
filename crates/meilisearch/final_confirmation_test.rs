// Final test to confirm the bug by looking at the actual bug scenario:
// The bug is NOT that documents with same distinct values are returned,
// but that the merge function doesnt apply distinct filtering AFTER merging
// So we need to test a scenario where both search types return valid results
// that would individually pass distinct filtering, but together violate it.

#[actix_rt::test]
async fn test_hybrid_merge_distinct_bug_confirmation() {
    let server = Server::new().await;
    let index = server.index("test");

    // Set up embedder
    let (response, code) = index
        .update_settings(json!({ 
            "embedders": {
                "default": {
                    "source": "userProvided",
                    "dimensions": 4
                }
            }
        }))
        .await;
    assert_eq!(202, code);
    index.wait_task(response.uid()).await.succeeded();

    // Create 6 documents with 3 different distinct values
    // We want both search types to return documents, then merge should apply distinct filtering
    let test_documents = json!([
        // Group 1: product_a
        {
            "id": 1,
            "title": "FIRST MATCHING TERMS QUERY SEARCH",  // High keyword relevance
            "description": "unrelated vector content",
            "product_id": "product_a",
            "_vectors": {"default": [0.1, 0.1, 0.1, 0.1]}  // Low vector relevance
        },
        {
            "id": 2,
            "title": "different unrelated content",  // Low keyword relevance
            "description": "but high vector similarity here",
            "product_id": "product_a",  // SAME distinct value
            "_vectors": {"default": [0.9, 0.9, 0.9, 0.9]}  // High vector relevance
        },
        // Group 2: product_b  
        {
            "id": 3,
            "title": "SECOND MATCHING TERMS QUERY SEARCH",
            "description": "another keyword match",
            "product_id": "product_b",
            "_vectors": {"default": [0.2, 0.2, 0.2, 0.2]}
        },
        {
            "id": 4,
            "title": "random stuff here",
            "description": "high vector match content",
            "product_id": "product_b",  // SAME distinct value
            "_vectors": {"default": [0.8, 0.8, 0.8, 0.8]}
        },
        // Group 3: product_c
        {
            "id": 5,
            "title": "THIRD MATCHING TERMS QUERY SEARCH",
            "description": "keyword relevance here",
            "product_id": "product_c",
            "_vectors": {"default": [0.3, 0.3, 0.3, 0.3]}
        },
        {
            "id": 6,
            "title": "unrelated keyword text",
            "description": "but vector similarity",
            "product_id": "product_c",  // SAME distinct value
            "_vectors": {"default": [0.7, 0.7, 0.7, 0.7]}
        }
    ]);

    let (response, code) = index.add_documents(test_documents, Some("id")).await;
    assert_eq!(202, code);
    index.wait_task(response.uid()).await.succeeded();

    // Set distinct on product_id
    let (task, _) = index.update_distinct_attribute(json!("product_id")).await;
    index.wait_task(task.uid()).await.succeeded();

    // Perform hybrid search that should match multiple documents
    let (response, code) = index
        .search_post(json!({
            "q": "MATCHING TERMS QUERY SEARCH",  // Should match docs 1, 3, 5 well
            "vector": [0.8, 0.8, 0.8, 0.8],     // Should match docs 2, 4, 6 well
            "hybrid": {
                "embedder": "default", 
                "semanticRatio": 0.5  // Equal weight
            },
            "limit": 10
        }))
        .await;

    assert_eq!(200, code);
    let hits = response["hits"].as_array().unwrap();
    
    println!("HYBRID SEARCH RESULTS (limit=10):");
    for (i, hit) in hits.iter().enumerate() {
        println!("  {}. id={}, product_id={}, title={}", 
                i+1, hit["id"], hit["product_id"], 
                hit["title"].as_str().unwrap_or("").chars().take(30).collect::<String>());
    }
    
    // Count distinct values
    let product_ids: Vec<&str> = hits.iter()
        .filter_map(|hit| hit["product_id"].as_str())
        .collect();
    
    let unique_product_ids: std::collections::HashSet<_> = product_ids.iter().collect();
    
    println!("Total results: {}, Unique product_ids: {}", product_ids.len(), unique_product_ids.len());
    println!("Product IDs found: {:?}", product_ids);
    
    // The bug would manifest as having more results than unique distinct values
    if product_ids.len() > unique_product_ids.len() {
        panic!("BUG REPRODUCED! Hybrid search returned {} documents but only {} unique distinct values. \
               This proves distinct filtering is not applied after merging vector and keyword results. \
               Product IDs: {:?}", product_ids.len(), unique_product_ids.len(), product_ids);
    }
    
    println!("Test completed. Expected up to 3 results (one per distinct value), got {}", hits.len());
    if hits.len() <= 3 {
        println!("Distinct filtering appears to be working correctly in this version");
    }
}
