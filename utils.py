from dataclasses import dataclass
import numpy as np 

def get_text_features(data):
    assert type(data) == str, 'Expect string'
    n_upper, n_lower, n_alpha, n_digits, n_spaces, n_numeric, n_special = 0, 0, 0, 0, 0, 0, 0
    number = 0
    specials_char = {
        '&': 0, '@': 1, '#': 2, '(': 3, ')': 4, '-': 5, '+': 6, 
        '=': 7, '*': 8, '%': 9, '.':10, ',': 11, '\\': 12,'/': 13, 
        '|': 14, ':': 15
    }
    specials_char_arr = np.zeros(shape=len(specials_char))

    for char in data:
        if char.islower():
            n_lower += 1
        if char.isupper():
            n_upper += 1
        if char.isspace():
            n_space += 1
        if char.isalpha():
            n_alpha += 1
        if char.isnumeric():
            n_numeric += 1
        if char in specials_char.keys():
            char_idx = specials_char[char]
            specials_char_arr[char_idx] += 1

    for word in data.split():
        try:
            number = int(word)
            n_digits += 1
        except:
            pass 

        if n_digits == 0:
            try:
                number = float(word)
                n_digits += 1
            except:
                pass 
    
    features = []
    features.append([
        n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_digits
    ])
    features = np.array(features)
    features = np.append(features, specials_char_arr)
    return features

if __name__=='__main__':
    data = '12.70'
    feature = get_text_features(data)
    print(feature)