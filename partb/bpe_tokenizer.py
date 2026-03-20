from heapq import heapify, heappop, heappush

class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        # raise NotImplementedError("BPETokenizer initialization not implemented yet.")
        self.vocab_size = vocab_size
        self.unk_id = None #set it in train
        self.merge_rules = [] #set it in train
        self.vocab = [] #same as before
        self.devocab = [] #same as before
        self.special_tokens = []

    def train(self, corpus):
        # raise NotImplementedError("Training method not implemented yet.")
        #need to initialize merge rules, vocab, devocab, 
        #merged_pair_heap = heaps of all pairs need to initialize this (occurence of a particular pair, pair)'s list
        merge_rules = []
        vocab = set()
        str_corpus = "".join(corpus)
        #corpus is a list of sentences, for each sentence, convert it to a string
        bigram_corpus = []
        for i in range(len(corpus)):
            sentence = corpus[i]
            sentence = list(sentence)
            bigram_corpus.extend([(sentence[i-1],sentence[i]) for i in range(1, len(sentence))])
            sent_vocab = set(sentence)
            vocab.union(sent_vocab)
        merged_pairs = {}
        for bigram in bigram_corpus:
            merged_pairs[bigram] = merged_pairs.get(bigram, 0) + 1
        merged_pairs = [(merged_pairs[bigram], bigram) for bigram in merged_pairs.keys()]
        merged_pairs_heap = heapify(merged_pairs)
        for i in range(self.vocab_size - len(vocab)):
            try:
                max_pair = heappop(merged_pairs_heap)
            except:
                break
            bigram_sep = max_pair[1]
            bigram = "".join(bigram_sep)
            merge_rules.append(bigram_sep)
            for token in vocab:
                word1 = token + bigram
                word2 = bigram + token
                freq1, freq2 = find_freq(str_corpus, word1, word2) #to be implemented
                if freq1 > 0:
                    merged_pairs_heap.heappush((freq1,(token, bigram)))
                if freq2 > 0:
                    merged_pairs_heap.heappush((freq2, (bigram, token)))
                    
        
            
        
    
    def encode(self, text):
        raise NotImplementedError("Encoding method not implemented yet.")

    def decode(self, token_ids):
        raise NotImplementedError("Decoding method not implemented yet.")

    def save(self, filepath):
        raise NotImplementedError("Save method not implemented yet.")

    def load(self, filepath):
        raise NotImplementedError("Load method not implemented yet.")
    
    def get_vocab_size(self):
        raise NotImplementedError("Get vocab size method not implemented yet.")
    
    def get_unk_id(self):
        raise NotImplementedError("Get unk id method not implemented yet.")
