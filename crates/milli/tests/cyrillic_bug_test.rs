// Test to validate the Cyrillic typo tolerance bug exists by demonstrating the problematic logic
#[test]
fn test_cyrillic_char_count_bug() {
    // Test the key insight from the RCA: word.len() vs word.chars().count()
    
    // ASCII word "doggy" (5 chars, 5 bytes)
    let ascii_word = "doggy";
    let ascii_byte_len = ascii_word.len();
    let ascii_char_count = ascii_word.chars().count();
    
    // Cyrillic word "собак" (5 chars, 10 bytes)  
    let cyrillic_word = "собак";
    let cyrillic_byte_len = cyrillic_word.len();
    let cyrillic_char_count = cyrillic_word.chars().count();
    
    eprintln!("ASCII '{}': byte_len={}, char_count={}", ascii_word, ascii_byte_len, ascii_char_count);
    eprintln!("Cyrillic '{}': byte_len={}, char_count={}", cyrillic_word, cyrillic_byte_len, cyrillic_char_count);
    
    // Simulate the buggy logic with default settings (oneTypo=5, twoTypos=9)
    let min_len_one_typo = 5;
    let min_len_two_typos = 9;
    
    // Current buggy implementation uses word.len() (byte count)
    let ascii_typos_buggy = if ascii_byte_len < min_len_one_typo {
        0
    } else if ascii_byte_len < min_len_two_typos {
        1
    } else {
        2
    };
    
    let cyrillic_typos_buggy = if cyrillic_byte_len < min_len_one_typo {
        0
    } else if cyrillic_byte_len < min_len_two_typos {
        1
    } else {
        2
    };
    
    eprintln!("Buggy logic (using byte count):");
    eprintln!("  ASCII '{}' gets {} typos", ascii_word, ascii_typos_buggy);
    eprintln!("  Cyrillic '{}' gets {} typos", cyrillic_word, cyrillic_typos_buggy);
    
    // Correct implementation should use word.chars().count()
    let ascii_typos_correct = if ascii_char_count < min_len_one_typo {
        0
    } else if ascii_char_count < min_len_two_typos {
        1
    } else {
        2
    };
    
    let cyrillic_typos_correct = if cyrillic_char_count < min_len_one_typo {
        0
    } else if cyrillic_char_count < min_len_two_typos {
        1
    } else {
        2
    };
    
    eprintln!("Correct logic (using character count):");
    eprintln!("  ASCII '{}' gets {} typos", ascii_word, ascii_typos_correct);
    eprintln!("  Cyrillic '{}' gets {} typos", cyrillic_word, cyrillic_typos_correct);
    
    // **THE BUG**: ASCII and Cyrillic should get same typos (both have 5 chars)
    // But buggy implementation gives them different typos - THIS IS THE FAILING ASSERTION
    assert_eq!(ascii_typos_buggy, cyrillic_typos_buggy,
        "BUG REPRODUCED: ASCII and Cyrillic words with same character count get different typo tolerance due to byte counting bug. ASCII: {}, Cyrillic: {}",
        ascii_typos_buggy, cyrillic_typos_buggy);
}
