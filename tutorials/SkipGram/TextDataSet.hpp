//#pragma once
#ifndef __TEXTDATASET__HPP
#define __TEXTDATASET__HPP    value

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <sstream>

#include "../../WICWIU_src/Tensor.hpp"
#include "../../WICWIU_src/DataLoader.hpp"

#include "TextDataSet/Field.hpp"
//#include "TextDataSet/TextDataSet_utils.hpp"
//#include "TextDataSet/ParalleledCorpusDataset.hpp"

template<typename DTYPE> class TextDataset : public Dataset<DTYPE>{ //전처리 옵션 관리하고
private:
  string path;
  char* m_pTextData;
  int text_length;
  int line_length;
  //-----Field 클래스에서 차용------//
  //옵션들

  Field* field;

  bool sequential = true;
  bool lower = true;
  bool padding = true;
  bool unk = true;
  //-----Vocab 클래스에서 차용------//
  map<int, string>* m_pIndex2Vocab;
  map<string, int>* m_pVocab2Frequency;
  map<string, int>* m_pVocab2Index;
  int n_vocabs;
  //-----넘겨주는 Data 관련!------//
  DTYPE **m_aaInput;
  DTYPE **m_aaLabel;

  int m_numOfInput;             //input data의 개수!!!
  int m_window;                 //window size -> 홀수가 기본이겠지!
  int m_negative;

  int m_dimOfInput;
  int m_dimOfLabel;

public:
  TextDataset(string path, Field field);

  virtual void                 Alloc(string path, Field field);

  void                         ReadFile();

  void                         Pad(); //아직!!!!

  void                         AddSentence(string sentence);

  void                         AddWord(string word);

  vector<string>               SplitBy(string input, char delimiter);

  string                       Preprocess(string sentence);

  string                       Preprocess(char* sentence);

  string                       Remove(string sentence, string delimiters);

  virtual void                 BuildVocab();

  virtual                      ~TextDataset();

  void                         Delete();

  int                          GetTextLength();

  void                         SetLineLength(int n);
  int                          GetLineLength();

  map<int, string>*            GetpIndex2Vocab();

  map<string, int>*            GetpVocab2Frequency();

  map<string, int>*            GetpVocab2Index();

  int                          GetNumberofVocabs();

  int                          GetNumberofWords();

  char*                        GetTextData();

  int                          GetInputDim();
  int                          GetLabelDim();
  void                         SetInputDim(int inputDim);
  void                         SetLabelDim(int labelDim);
  //virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

};


template<typename DTYPE> map<int, string>* TextDataset<DTYPE>::GetpIndex2Vocab(){
  return m_pIndex2Vocab;
}

template<typename DTYPE> map<string, int>* TextDataset<DTYPE>::GetpVocab2Frequency(){
  return m_pVocab2Frequency;
}

template<typename DTYPE> map<string, int>* TextDataset<DTYPE>::GetpVocab2Index(){
  return m_pVocab2Index;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetNumberofWords(){
  map<string, int>::iterator it;
  int result = 0;

  for(it=m_pVocab2Frequency->begin(); it!=m_pVocab2Frequency->end(); it++){
    result += it->second;
  }
  return result;
}

template<typename DTYPE> int TextDataset<DTYPE>::GetNumberofVocabs(){
  return n_vocabs-1;
}

template<typename DTYPE> char* TextDataset<DTYPE>::GetTextData(){
  return m_pTextData;
}

template<typename DTYPE> TextDataset<DTYPE>::TextDataset(string path, Field field) {

  cout<<"<<<<<<<<<<<<< TextDataset Constructor >>>>>>>>>>>>>>>"<<endl;
  this->path="";
  text_length = 0;
  line_length = 0;

  this->field = NULL;
  m_pIndex2Vocab = NULL;
  m_pVocab2Frequency = NULL;
  m_pVocab2Index = NULL;
  n_vocabs = 0;

  m_aaInput = NULL;
  m_aaLabel = NULL;
  m_numOfInput = 0;
  m_window     = 0;
  m_negative   = 0;
  m_dimOfInput = 0;
  m_dimOfLabel = 0;

  Alloc(path, field);
}


template<typename DTYPE> void TextDataset<DTYPE>::Alloc(string path, Field field) {

  this->path = path;
  this->field = new Field(field);
  m_pIndex2Vocab = new map<int, string>();
  m_pVocab2Frequency = new map<string, int>();
  m_pVocab2Index = new map<string, int>();

  AddWord("<s>");
  AddWord("<e>");


  ReadFile();
}

template<typename DTYPE> TextDataset<DTYPE>::~TextDataset() {
  cout << "TextDataset 소멸자 호출" << endl;
  Delete();
}

template<typename DTYPE> void TextDataset<DTYPE>::Delete() {
  if(m_pTextData) {
    delete[] m_pTextData;
    m_pTextData = NULL;
  }

  if(m_pIndex2Vocab) {
    delete m_pIndex2Vocab;
    m_pIndex2Vocab = NULL;
  }

  if(m_pVocab2Frequency) {
    delete m_pVocab2Frequency;
    m_pVocab2Frequency = NULL;
  }

  if(m_pVocab2Index != NULL) {
    delete m_pVocab2Index;
    m_pVocab2Index = NULL;
  }

  if(field != NULL) {
    delete field;
    field = NULL;
  }
}

template<typename DTYPE> void TextDataset<DTYPE>::ReadFile() {
  field->makeSequential();

  cout<<"<<<<<<<<<<<<<<<<  FileReader  >>>>>>>>>>>>>>>>>>>>"<<endl;
    this->path = path;
    cout << this->path << endl;
    ifstream fin;
    fin.open(path);

    if(fin.is_open()) {

      fin.seekg(0, ios::end);
      text_length = fin.tellg();
      fin.tellg();
      fin.seekg(0, ios::beg);

      m_pTextData = new char[text_length];
      //파일 읽기
      fin.read(m_pTextData, text_length);

      text_length = strlen(m_pTextData);
      fin.close();
    }

    else {
      cout<<"ERROR : CANNOT OPEN FILE"<<endl;
      cout<<"PATH : "<<path<<endl;         //파일 없거나 안열릴시
      exit(-1);
    }
    //cout<<text_length<<endl;
}

template<typename DTYPE> void TextDataset<DTYPE>::AddSentence(string sentence){
  //cout<<"<<<<<<<<<<<<<<<<  AddSentence  >>>>>>>>>>>>>>>>>>>>"<<endl;
  vector<string> words = SplitBy(sentence, ' ');
  for(string word: words){
    AddWord(word);
  }
  vector<string>().swap(words);
}

template<typename DTYPE> void TextDataset<DTYPE>::AddWord(string word){
  if(m_pVocab2Index->find(word)==m_pVocab2Index->end()){
    m_pVocab2Index->insert(make_pair(word, n_vocabs));
    m_pVocab2Frequency->insert(make_pair(word, 1));
    m_pIndex2Vocab->insert(make_pair(n_vocabs, word));
    n_vocabs ++;
  }
  else{
    m_pVocab2Frequency->at(word)++;
  }
}

template<typename DTYPE> vector<string> TextDataset<DTYPE>::SplitBy(string input, char delimiter) {
  vector<string> answer;
  stringstream ss(input);
  string temp;

  while (getline(ss, temp, delimiter)) {
      answer.push_back(temp);
  }
  return answer;
}



template<typename DTYPE> string TextDataset<DTYPE>::Preprocess(string sentence) {
  if(lower){
    transform(sentence.begin(), sentence.end(), sentence.begin(), [](unsigned char c){ return std::tolower(c); });
  }
  sentence = Remove(sentence, ",.?!\"\'><:-");
  return sentence;
}

template<typename DTYPE> string TextDataset<DTYPE>::Preprocess(char* sentence){
  string new_sentence(sentence);
  return Preprocess(new_sentence);
}

template<typename DTYPE> string TextDataset<DTYPE>:: Remove(string str, string delimiters){
  vector<string> splited_delimiters;
  for(int i=0; i<delimiters.length(); i++){
    splited_delimiters.push_back(delimiters.substr(i,1));
  }
  for(string delimiter : splited_delimiters){
    int k = str.find(delimiter);
    while(k>=0){
      string k_afterStr = str.substr(k+1, str.length()-k);
      str = str.erase(k) + k_afterStr;
      k = str.find(delimiter);
    }
  }
    return str;
}

template<typename DTYPE> void TextDataset<DTYPE>:: BuildVocab(){
    cout<<"<<<<<<<<<<<<<<<<  BuildVocab 호출 >>>>>>>>>>>>>>>>>>>>"<<endl;
};

template<typename DTYPE> int TextDataset<DTYPE>:: GetTextLength(){
  return text_length;
}
template<typename DTYPE> int TextDataset<DTYPE>:: GetLineLength(){
  return line_length;
}
template<typename DTYPE> void TextDataset<DTYPE>:: SetLineLength(int n){
  line_length = n;
}
template<typename DTYPE> int TextDataset<DTYPE>::GetInputDim(){
    return m_dimOfInput;
}
template<typename DTYPE> int TextDataset<DTYPE>::GetLabelDim(){
    return m_dimOfLabel;
}
template<typename DTYPE> void TextDataset<DTYPE>::SetInputDim(int inputDim){
    m_dimOfInput = inputDim;
}
template<typename DTYPE> void TextDataset<DTYPE>::SetLabelDim(int labelDim){
    m_dimOfLabel = labelDim;
}

#endif