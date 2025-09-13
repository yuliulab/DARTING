#! /user/bin/python
#Author: yingwang

class SmilesCharDictionary(object):
    """ 
    A fixed dictionary for druglike SMILES
    convert smile to  token
    a spcae:0 for padding,Q:1 as the start token and
    end_of_line \n:2 as the stop token
    """
    PAD = ""
    BEGIN = "Q"
    END="\n"

    # set parameters
    def __init__(self,max_len=120) -> None:
        #define  forbidden_symbols
        self.forbidden_symbols = {
        'Ag', 'Al', 'Am', 'Ar', 'At', 'Au', 'D', 'E', 'Fe', 'G', 'K', 'L', 'M', 'Ra', 'Re',
        'Rf', 'Rg', 'Rh', 'Ru', 'T', 'U', 'V', 'W', 'Xe',
        'Y', 'Zr', 'a', 'd', 'f', 'g', 'h', 'k', 'm', 'si', 't', 'te', 'u', 'v', 'y'}
        #define dict for the element of the smiles
        self.char_idx={
        self.PAD: 0, self.BEGIN: 1, self.END: 2, '#': 20, '%': 22, '(': 25, ')': 24, '+': 26, '-': 27,
        '.': 30,
        '0': 32, '1': 31, '2': 34, '3': 33, '4': 36, '5': 35, '6': 38, '7': 37, '8': 40,
        '9': 39, '=': 41, 'A': 7, 'B': 11, 'C': 19, 'F': 4, 'H': 6, 'I': 5, 'N': 10,
        'O': 9, 'P': 12, 'S': 13, 'X': 15, 'Y': 14, 'Z': 3, '[': 16, ']': 18,
        'b': 21, 'c': 8, 'n': 17, 'o': 29, 'p': 23, 's': 28,
        "@": 42, "R": 43, '/': 44, "\\": 45, 'E': 46
        }
        # a dict for id to char
        self.idx_char = {v:k for k,v in self.char_idx.items()}

        #define a dict for convert complex element in the smile to a simple chr
        self.encode_dict = {"Br": 'Y', "Cl": 'X', "Si": 'A', 'Se': 'Z', '@@': 'R', 'se': 'E'}
        self.decode_dict = {v:k for k,v in self.encode_dict.items()}
    
    #define illigal symbol
    def allowed(self,smiles)->bool:
        """
        smiles: SMILE string
        return: True if all legal
        """
        # judgement:
        for symbol in self.forbidden_symbols:
            if symbol in smiles:
                print("Forbidden symbol {:<2} in {}".format(symbol,smiles))
                return False
            return True
    def encode(self,smiles:str)->str:
        """repalce multi-char token with single token in SMILES string 
        eg, '@@' to 'R' 
        Args:
        smiles: SMILE string

        Return: standered smile string with onlg single-char token
        """
        temp_smiles = smiles

        for symbol,token in self.encode_dict.items():
            temp_smiles = temp_smiles.replace(symbol,token)
        return temp_smiles
    
    def decode(self,smiles):
        '''
        replace special token to multi-char
        Args:
        smiles: SMILE string

        return: a smile sting possibly multi-char
        '''
        temp_smiles = smiles
        for symbol,token in self.decode_dict.items():
            temp_smiles = temp_smiles.replace(symbol,token)
        return temp_smiles
    
    def get_cahr_num(self)-> int:
        """
        return : the number of the characters in the alphabet
        """

        return len(self.idx_cahr)
    
    @property
    def begin_idx(self)->int:
        return self.char_idx[self.BEGIN]
    
    @property
    def end_idx(self)-> int:
        return self.char_idx[self.END]
    
    @property
    def pad_idx(self)->int:
        return self.char_idx[self.PAD]
    
    def matirx_to_smiles(self,array):
        """
        convert an matix of indices to their smiles representations
        Args:
        array: torch tensor od indicies , one molecule per row

        Return: a list of smiles, without the termination symbol
        """
        smiles_strings = []
        for row in array:
            predicted_chars = []
            for j in row:
                # j is id, convert id to char
                next_char = self.idx_char[j.item()]
                if next_char == self.END:
                    break
                predicted_chars.append(next_char)
            
            smi = ''.join(predicted_chars)
            smi = self.decode(smi)
            smiles_strings.append(smi)
        return smiles_strings

