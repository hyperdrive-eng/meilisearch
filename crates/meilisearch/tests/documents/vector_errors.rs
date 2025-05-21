use meili_snap::*;

use crate::common::Server;
use crate::json;

#[actix_rt::test]
async fn test_vectors_array_instead_of_map() {
    let server = Server::new().await;
    let index = server.index("test");
    
    // Add a document with _vectors field as an array instead of a map
    let document = json!([{
        "id": 1,
        "title": "Test document",
        "_vectors": [0.1, 0.2, 0.3]  // Error: should be a map, not an array
    }]);
    
    let (response, code) = index.add_documents(document, None).await;
    
    // This should return a 400 Bad Request, but currently returns a 500 Internal Server Error
    // Once the bug is fixed, this test will fail and need to be updated to expect 400
    snapshot!(code, @"500 Internal Server Error");
    
    // Check that the error message mentions the vector format
    snapshot!(json_string!(response), @r###"
    {
      "message": "An internal error has occurred.",
      "code": "internal",
      "type": "internal",
      "link": "https://docs.meilisearch.com/errors#internal"
    }
    "###);
}
