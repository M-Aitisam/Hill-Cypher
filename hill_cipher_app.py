import numpy as np
import streamlit as st
from math import gcd
from sympy import Matrix

def text_to_num(text):
    """Convert text to numerical values (A=0, B=1, ..., Z=25)"""
    return [ord(char) - ord('A') for char in text]

def num_to_text(nums):
    """Convert numerical values back to text"""
    return ''.join([chr(num + ord('A')) for num in nums])

def pad_plaintext(plaintext, block_size=2):
    """Pad plaintext with 'X' if length not multiple of block_size"""
    padding_length = (block_size - len(plaintext) % block_size) % block_size
    return plaintext + 'X' * padding_length

def matrix_mod_inv(matrix, modulus):
    """Find the modular inverse of a matrix"""
    det = int(np.round(np.linalg.det(matrix)))
    det_inv = pow(det, -1, modulus)
    adjugate = np.round(det * np.linalg.inv(matrix)).astype(int)
    return (det_inv * adjugate) % modulus

def is_valid_key(key, modulus=26):
    """Check if key matrix is invertible modulo modulus"""
    try:
        det = int(np.round(np.linalg.det(key)))
        return gcd(det, modulus) == 1
    except:
        return False

def hill_encrypt(plaintext, key):
    """Encrypt plaintext using Hill Cipher"""
    block_size = key.shape[0]
    plaintext = pad_plaintext(plaintext, block_size)
    numerical = text_to_num(plaintext)
    ciphertext = []
    
    for i in range(0, len(numerical), block_size):
        block = numerical[i:i+block_size]
        encrypted_block = np.dot(key, block) % 26
        ciphertext.extend(encrypted_block)
    
    return num_to_text(ciphertext)

def hill_decrypt(ciphertext, key):
    """Decrypt ciphertext using Hill Cipher"""
    block_size = key.shape[0]
    numerical = text_to_num(ciphertext)
    plaintext = []
    
    try:
        key_inv = matrix_mod_inv(key, 26)
    except ValueError as e:
        raise ValueError("Cannot decrypt - key matrix is not invertible") from e
    
    for i in range(0, len(numerical), block_size):
        block = numerical[i:i+block_size]
        decrypted_block = np.dot(key_inv, block) % 26
        plaintext.extend(decrypted_block)
    
    return num_to_text(plaintext)

def main():
    st.title("Hill Cipher Encryption/Decryption")
    st.markdown("""
    This app implements the Hill Cipher algorithm for encrypting and decrypting text messages.
    The cipher uses matrix multiplication modulo 26 for encryption and decryption.
    """)
    
    st.header("Example from Question")
    st.markdown("""
    **Plaintext:** "HELP"  
    **Key Matrix:**  
    ```
    [3, 3]
    [2, 5]
    ```
    """)
    
    if st.button("Run Example"):
        example_key = np.array([[3, 3], [2, 5]])
        plaintext = "HELP"
        
        # Encryption
        ciphertext = hill_encrypt(plaintext, example_key)
        st.success(f"Encrypted ciphertext: {ciphertext}")
        
        # Key inverse
        try:
            key_inv = matrix_mod_inv(example_key, 26)
            st.latex(f"K^{{-1}} = {key_inv.tolist()}")
            
            # Verify K⁻¹ × K = I
            identity = np.dot(key_inv, example_key) % 26
            st.latex(f"K^{{-1}} \\times K = {identity.tolist()} (Identity matrix)")
        except ValueError as e:
            st.error(str(e))
        
        # Decryption
        try:
            decrypted = hill_decrypt(ciphertext, example_key)
            st.success(f"Decrypted plaintext: {decrypted}")
        except ValueError as e:
            st.error(str(e))
    
    st.header("Custom Encryption/Decryption")
    tab1, tab2 = st.tabs(["Encrypt", "Decrypt"])
    
    with tab1:
        st.subheader("Encryption")
        plaintext = st.text_input("Plaintext (A-Z only, no spaces)", "HELP", key="encrypt_plaintext").upper()
        key_input = st.text_area("Key Matrix (e.g., '3 3\n2 5' for 2x2 matrix)", "3 3\n2 5", key="encrypt_key")
        
        try:
            key_rows = [list(map(int, row.split())) for row in key_input.split('\n') if row.strip()]
            key = np.array(key_rows)
            
            if st.button("Encrypt", key="encrypt_button"):
                if not plaintext.isalpha():
                    st.error("Plaintext must contain only letters A-Z")
                elif not is_valid_key(key):
                    st.error("Key matrix is not invertible modulo 26")
                else:
                    ciphertext = hill_encrypt(plaintext, key)
                    st.success(f"Ciphertext: {ciphertext}")
                    
                    # Show details
                    st.subheader("Encryption Details")
                    padded = pad_plaintext(plaintext, key.shape[0])
                    st.write(f"Padded plaintext: {padded}")
                    st.write(f"Numerical plaintext: {text_to_num(padded)}")
                    st.write(f"Key matrix:\n{key}")
        except Exception as e:
            st.error(f"Invalid key matrix format: {e}")
    
    with tab2:
        st.subheader("Decryption")
        ciphertext = st.text_input("Ciphertext (A-Z only, no spaces)", "", key="decrypt_ciphertext").upper()
        key_input = st.text_area("Key Matrix for decryption", "3 3\n2 5", key="decrypt_key")
        
        try:
            key_rows = [list(map(int, row.split())) for row in key_input.split('\n') if row.strip()]
            key = np.array(key_rows)
            
            if st.button("Decrypt", key="decrypt_button"):
                if not ciphertext.isalpha():
                    st.error("Ciphertext must contain only letters A-Z")
                elif not is_valid_key(key):
                    st.error("Key matrix is not invertible modulo 26")
                else:
                    try:
                        plaintext = hill_decrypt(ciphertext, key)
                        st.success(f"Decrypted plaintext: {plaintext}")
                        
                        # Show details
                        st.subheader("Decryption Details")
                        st.write(f"Numerical ciphertext: {text_to_num(ciphertext)}")
                        key_inv = matrix_mod_inv(key, 26)
                        st.write(f"Inverse key matrix:\n{key_inv}")
                    except ValueError as e:
                        st.error(str(e))
        except Exception as e:
            st.error(f"Invalid key matrix format: {e}")
    
    st.header("Algorithm Explanation")
    with st.expander("How the Hill Cipher works"):
        st.markdown("""
        ### Encryption Process:
        1. **Convert letters to numbers**: A=0, B=1, ..., Z=25
        2. **Pad the plaintext** if length isn't multiple of key matrix size
        3. **Split into vectors** of size matching key matrix dimensions
        4. **Multiply each vector by key matrix** modulo 26
        5. **Convert numbers back to letters** to get ciphertext

        ### Decryption Process:
        1. **Find modular inverse** of key matrix (mod 26)
        2. **Convert ciphertext letters to numbers**
        3. **Split into vectors** of appropriate size
        4. **Multiply each vector by inverse key matrix** modulo 26
        5. **Convert numbers back to letters** to recover plaintext

        ### Why invertible key matrix?
        The key matrix must be invertible modulo 26 to ensure decryption is possible.
        This means its determinant must be coprime with 26 (gcd(det(K), 26) = 1).
        """)

if __name__ == "__main__":
    main()