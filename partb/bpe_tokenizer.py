from heapq import heapify, heappop, heappush
import json
import os

def find_freq(str_corpus, word):
    freq = 0
    for i in range(len(str_corpus) - len(word) + 1):
        if word == str_corpus[i:i+len(word)]:
            freq+=1
    return freq

def merge_data(t1, t2):
    return t1+t2

class Node:
    def __init__(self, data = -1, next = None, prev = None):
        self.data = data
        self.next = next
        self.prev = prev

class DLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, data):
        if self.head == None:
            self.head = Node(data)
            self.tail = self.head
        else:
            self.tail.next = Node(data, prev = self.tail)
            self.tail = self.tail.next
    
    def merge(self, node, merge_data):  #merge node and its next node to form a single node
        if node.next != None:
            data1 = node.data
            data2 = node.next.data
            new_data = merge_data(data1, data2)
            new_node = Node(new_data, prev=node.prev, next=node.next.next)
            if self.head == node:
                self.head = new_node
            else:
                node.prev.next = new_node
            if self.tail == node.next:
                self.tail = new_node
            else:
                node.next.next.prev = new_node
            return new_node
            

class BPETokenizer:
    def __init__(self, vocab_size=15000, special_tokens=None):
        # raise NotImplementedError("BPETokenizer initialization not implemented yet.")
        self.vocab_size = vocab_size
        self.unk_id = None #set it in train
        self.merge_rules = [] #set it in train
        self.vocab = [] #same as before
        self.devocab = {} #same as before
        self.special_tokens = special_tokens

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
            vocab.update(sent_vocab)
        #have a doubly linked list of the corpus and then merge the pairs in the corpus and then update the
        #merge frequencies appropriately
        DLlist = DLinkedList()
        for char in str_corpus:
            DLlist.append(char)

        merged_pairs = {}
        for bigram in bigram_corpus:
            merged_pairs[bigram] = merged_pairs.get(bigram, 0) + 1
        merged_pairs_heap = [(-merged_pairs[bigram], bigram) for bigram in merged_pairs.keys()]
        heapify(merged_pairs_heap)
        valid_counts = dict(merged_pairs)
        for i in range(self.vocab_size - len(vocab)):
            max_pair = None
            while merged_pairs_heap:
                neg_freq, bigram = heappop(merged_pairs_heap)
                if valid_counts.get(bigram, 0) == -neg_freq:
                    max_pair = (neg_freq, bigram)
                    break
            if max_pair is None:
                break
            bigram_sep = max_pair[1]
            bigram = "".join(bigram_sep)
            vocab.add(bigram)
            merge_rules.append(bigram_sep)
            new_freq = {}
            node = DLlist.head
            while node!=DLlist.tail and node != None:
                if node.data == bigram_sep[0] and node.next != None and node.next.data == bigram_sep[1]:
                    new_node = DLlist.merge(node, merge_data)
                    if new_node.prev != None:
                        new_bigram1 = (new_node.prev.data, new_node.data)
                        new_freq[new_bigram1] = new_freq.get(new_bigram1, 0) + 1
                    if new_node.next != None:
                        new_bigram2 = (new_node.data, new_node.next.data)
                        new_freq[new_bigram2] = new_freq.get(new_bigram2, 0) + 1
                    #for the removed bigrams, we need to check the previous and next nodes of the old nodes
                    if node.prev != None:
                        old_bigram1 = (node.prev.data, node.data)
                        new_freq[old_bigram1] = new_freq.get(old_bigram1, 0) - 1
                    if node.next != None and node.next.next != None:
                        old_bigram2 = (node.next.data, node.next.next.data)
                        new_freq[old_bigram2] = new_freq.get(old_bigram2, 0) - 1
                    node = new_node
                node = node.next
                # print()
            for bigram, delta in new_freq.items():
                valid_counts[bigram] = valid_counts.get(bigram, 0) + delta
                if valid_counts[bigram] > 0:
                    heappush(merged_pairs_heap, (-valid_counts[bigram], bigram))
            # for token in vocab:
            #     word1 = token + bigram
            #     word2 = bigram + token
            #     # freq1, freq2 = find_freq(str_corpus, word1, word2) #to be implemented
            #     freq1 = find_freq(str_corpus, word1)
            #     freq2 = find_freq(str_corpus, word2)
            #     if freq1 > 0:
            #         # merged_pairs_heap.heappush((-freq1,(token, bigram)))
            #         heappush(merged_pairs_heap, (-freq1,(token, bigram)))
            #     if freq2 > 0:
            #         # merged_pairs_heap.heappush((-freq2, (bigram, token)))
            #         heappush(merged_pairs_heap, (-freq2,(bigram, token)))
        # self.vocab = list(vocab)
        pad_token = "<PAD>"
        vocab = list(vocab)
        vocab = sorted(vocab)         # deterministic ordering
        vocab.insert(0, pad_token)    # 0 is always padding, never a real token
        self.vocab = vocab
        self.merge_rules = merge_rules
        #add unknown and special things
        self.vocab.append("<UNK>")
        self.unk_id = len(self.vocab) - 1
        # special_tokens = set(self.special_tokens)
        # self.vocab.extend(special_tokens)
        if self.special_tokens != None:
            if isinstance(self.special_tokens, list):
                self.vocab.extend(self.special_tokens)
            else:
                self.vocab.append(self.special_tokens)
        
        #set devocab
        self.devocab = {}
        for i in range(len(self.vocab)):
            self.devocab[self.vocab[i]] = i
                    
        
            
        
    
    def encode(self, text):
        # raise NotImplementedError("Encoding method not implemented yet.")
        #text is a single string - line of text 
        #need to split it up into character level first 
        #need to iterate thru the merge rules and combine whatever can be combined
        #nextiterate thru the text adn convert them to ids based on devocab
        #return the list of ids
        chars = list(text)
        DLlist = DLinkedList()
        for char in chars:
            DLlist.append(char)
        for rule in self.merge_rules:
            node = DLlist.head
            while node!=DLlist.tail and node != None:
                if tuple(rule) == (node.data, node.next.data):
                    node = DLlist.merge(node, merge_data)
                node = node.next
        token_ids = []
        node = DLlist.head
        while node!=None:
            token_ids.append(self.devocab.get(node.data, self.unk_id))
            node = node.next
        return token_ids
        pass

    def decode(self, token_ids):
        # raise NotImplementedError("Decoding method not implemented yet.")
        #you will get a list of ints in token ids
        #conver it into a listof str
        #convert that back into a single text by "".join(list)
        list_of_tokens = list(map(lambda x: self.vocab[x], token_ids))
        # print(token_ids, self.vocab, self.devocab)
        return "".join(list_of_tokens)

    def save(self, filepath):
        # raise NotImplementedError("Save method not implemented yet.")
        filename = os.path.join(filepath, "tokenizer.json")
        with open(filename, "w") as fh:
            json.dump(self.__dict__, fh)
        pass

    def load(self, filepath):
        # raise NotImplementedError("Load method not implemented yet.")
        filename = os.path.join(filepath, "tokenizer.json")
        with open(filename, "r") as fh:
            tokeniser = json.load(fh)
            self.__dict__.update(tokeniser)
        pass
    
    def get_vocab_size(self):
        # raise NotImplementedError("Get vocab size method not implemented yet.")
        # return self.vocab_size
        return len(self.vocab)
    
    def get_unk_id(self):
        # raise NotImplementedError("Get unk id method not implemented yet.")
        return self.unk_id
