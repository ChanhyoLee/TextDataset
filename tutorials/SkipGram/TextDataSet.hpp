//#pragma once

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
  char* TextData;
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
  // //-----넘겨주는 Data 관련!------//
  // DTYPE **m_aaInput;
  // DTYPE **m_aaLabel;
  //
  // int m_numOfInput;             //input data의 개수!!!
  // int m_window;                 //window size -> 홀수가 기본이겠지!
  // int m_negative;
  //
  // int m_dimOfInput;
  // int m_dimOfLabel;

public:
  TextDataset(string path, Field field);

  void                         Alloc(string path, Field field);

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
  return TextData;
}

template<typename DTYPE> TextDataset<DTYPE>::TextDataset(string _path, Field field) {

  cout<<"<<<<<<<<<<<<< TextDataset Constructor >>>>>>>>>>>>>>>"<<endl;
  path="";
  text_length = 0;
  line_length = 0;

  this->field = NULL;
  m_pIndex2Vocab = NULL;
  m_pVocab2Frequency = NULL;
  m_pVocab2Index = NULL;
  n_vocabs = 0;

  // m_aaInput = NULL;
  // m_aaLabel = NULL;
  // m_numOfInput = 0;
  // m_window     = 0;
  // m_negative   = 0;
  // m_dimOfInput = 0;
  // m_dimOfLabel = 0;

  Alloc(_path, field);
}


template<typename DTYPE> void TextDataset<DTYPE>::Alloc(string _path, Field field) {

  path = _path;
  this->field = new Field(field);
  m_pIndex2Vocab = new map<int, string>();
  m_pVocab2Frequency = new map<string, int>();
  m_pVocab2Index = new map<string, int>();

  ReadFile();
}

template<typename DTYPE> TextDataset<DTYPE>::~TextDataset() {
  cout << "TextDataset 소멸자 호출" << endl;
  Delete();
}

template<typename DTYPE> void TextDataset<DTYPE>::Delete() {
  if(TextData) {
    delete[] TextData;
    TextData = NULL;
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

      TextData = new char[text_length];
      //파일 읽기
      fin.read(TextData, text_length);

      text_length = strlen(TextData);
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

// template<typename DTYPE> std::vector<Tensor<DTYPE> *>* TextDataset<DTYPE>:: GetData(int idx){
//   std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

//   Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfInput);
//   Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfLabel);

//   for (int i = 0; i < m_dimOfInput; i++) {
//       //이거는 전체 단어의 개수 안 맞춰주면 이렇게 됨!!!
//       if(m_aaInput[idx][i]==-1)
//           std::cout<<'\n'<<"****************************************************************************************음수존재..."<<'\n';
//       (*input)[i] = m_aaInput[idx][i];
//   }

//   //(*label)[ (int)m_aaLabel[idx][0] ] = 1.f;
//   (*label)[0] = 1.f;

//   result->push_back(input);
//   result->push_back(label);

//   return result;
// }


//--------------------------------------------------병렬 코퍼스 데이터--------------------------------------------------//
// template<typename DTYPE>
// class ParalleledCorpusDataset : public TextDataset<DTYPE>{ //파일 경로 받아서 실제 보캡, Paired문장 등 보관
// private:
//   pair<string, string> m_languageName;
//   vector< pair<string, string> >* m_pairedSentences;          // paired data
//   vector< pair< int*, int* > >* m_pairedIndexedSentences;
// public:
//   ParalleledCorpusDataset(string path, string srcName, string dstName, Field field);
//
//   void                                   Alloc();
//
//   void                                   MakeLineData();
//
//   virtual void                           BuildVocab();
//
//   virtual                                ~ParalleledCorpusDataset();
//
//   void                                   Delete();
//
//
//   vector< pair<string, string> >*        GetPairedSentences();
//
//   vector< pair< int*, int* > >*          GetmPairedIndexedSentences();
//   //virtual std::vector<Tensor<DTYPE>*>*   GetData(int idx);
// };
//
//
// template<typename DTYPE> ParalleledCorpusDataset<DTYPE>::ParalleledCorpusDataset(string path, string srcName, string dstName, Field field) : TextDataset<DTYPE>::TextDataset(path, field) {
//
//   m_languageName = make_pair(srcName, dstName);
//
//   m_pairedSentences = NULL;
//   m_pairedIndexedSentences = NULL;
//
//   Alloc();
// }
// template<typename DTYPE> ParalleledCorpusDataset<DTYPE>::~ParalleledCorpusDataset() {
//     cout << "ParalleledCorpusDataset 소멸자 호출" << endl;
//     Delete();
// }
//
// template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::Delete() {
//   if(m_pairedSentences) {
//     delete m_pairedSentences;
//     m_pairedSentences = NULL;
//   }
//
//   if(m_pairedIndexedSentences) {
//     delete m_pairedIndexedSentences;
//     m_pairedIndexedSentences = NULL;
//   }
// }
//
//
// template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::Alloc() {
//
//   m_pairedSentences = new vector< pair<string, string> >();
//   m_pairedIndexedSentences = new vector< pair< int*, int* > >();
// }
//
// template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::BuildVocab() {
//   MakeLineData();
//   cout<<"<<<<<<<<<<<<<<<<  BuildVocab 호출 >>>>>>>>>>>>>>>>>>>>"<<endl;
//   //cout << m_pairedSentences->size() << endl;
//   for(int i=0; i<m_pairedSentences->size(); i++){
//     vector<string> temp_words = this->SplitBy(m_pairedSentences->at(i).first, ' ');
//     vector<int> temp_first_indexed_words;
//     for(string word: temp_words){
//       //cout << word << endl;
//       temp_first_indexed_words.push_back(this->GetpVocab2Index()->at(word));
//     }
//     temp_words = this->SplitBy(m_pairedSentences->at(i).second, ' ');
//     vector<int> temp_second_indexed_words;
//     for(string word: temp_words){
//       temp_second_indexed_words.push_back(this->GetpVocab2Index()->at(word));
//     }
//     pair<int*, int*> temp_pair = make_pair(&temp_first_indexed_words[0], &temp_second_indexed_words[0]);
//     m_pairedIndexedSentences->push_back(temp_pair);
//   }
//   m_pairedIndexedSentences->shrink_to_fit();
// }
//
// template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::MakeLineData() { // 확인완료
//
//     cout<<"<<<<<<<<<<<<<<<<  MakeLineData  >>>>>>>>>>>>>>>>>>>>"<<endl;
//     //cout<<strlen(TextData)<<endl;
//     char* token = strtok(this->GetTextData(), "\t\n");
//     char* last_sentence = NULL;
//
//     while(token != NULL) {
//       //cout<<token<<endl;              //DEBUG
//       if(this->GetLineLength()%2==0){
//         last_sentence = token;                                              // paired data를 만들기위해 앞에 오는 line 임시 저장
//       }
//       else {
//         string str_last_sentence = this->Preprocess(last_sentence);
//         string str_token = this->Preprocess(token);
//         m_pairedSentences->push_back(make_pair(str_last_sentence, str_token));           // paired data 저장
//         this->AddSentence(this->Preprocess(m_pairedSentences->back().first));
//         this->AddSentence(this->Preprocess(m_pairedSentences->back().second));
//       }
//       //temp->line->push_back(token);                                         // 각 언어에 line 저장
//       //MakeVocab(token);
//       token = strtok(NULL, "\t\n");
//       int temp_lineLength = this->GetLineLength();
//       if(temp_lineLength%10000==0)
//         cout<<"line_length = "<<temp_lineLength<<endl;
//
//       this->SetLineLength(++temp_lineLength);
//     }
//     m_pairedSentences->shrink_to_fit();
//     //text_lines /=2;
//   }
//
//
// template<typename DTYPE>  vector< pair<string, string> >* ParalleledCorpusDataset<DTYPE>::GetPairedSentences() {
//   return m_pairedSentences;
// }
//
// template<typename DTYPE>  vector< pair< int*, int* > >* ParalleledCorpusDataset<DTYPE>::GetmPairedIndexedSentences() {
//   return m_pairedIndexedSentences;
// }



// template<typename DTYPE> std::vector<Tensor<DTYPE> *>* ParalleledCorpusDataset<DTYPE>:: GetData(int idx){
//   std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

//   Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfInput);
//   Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfLabel);

//   for (int i = 0; i < m_dimOfInput; i++) {
//       //이거는 전체 단어의 개수 안 맞춰주면 이렇게 됨!!!
//       if(m_aaInput[idx][i]==-1)
//           std::cout<<'\n'<<"****************************************************************************************음수존재..."<<'\n';
//       (*input)[i] = m_aaInput[idx][i];
//   }

//   //(*label)[ (int)m_aaLabel[idx][0] ] = 1.f;
//   (*label)[0] = 1.f;

//   result->push_back(input);
//   result->push_back(label);

//   return result;
// }

// int main(){
//   ParalleledCorpusDataset<float>* translation_data = new ParalleledCorpusDataset<float>("eng-fra.txt", "eng", "fra");

//   translation_data->BuildVocab();
//   cout << "LineLength:  " << translation_data->GetLineLength() << endl;
//   cout << "TextLength:  " << translation_data->GetTextLength() << endl;
//   cout << "NumofWords:  " << translation_data->GetNumberofWords() << endl;
//   cout << "NumofVocabs: " << translation_data->GetNumberofVocabs() << endl;

//   map<int, string> *index2vocab = translation_data->GetpIndex2Vocab();
//   map<int, string> :: iterator iter;
//   int count = 0;
//   for ( iter = index2vocab->begin(); iter != index2vocab->end(); iter++ ){
//     cout << iter->first << " : " << iter->second << "\t";
//     if(count%5==0){
//       cout << endl;
//     }
//     count ++;
//   }
// }
