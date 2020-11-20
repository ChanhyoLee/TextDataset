#ifndef __PARALLELED_CORPUS_DATASET_HPP__
#define __PARALLELED_CORPUS_DATASET_HPP__

#include "../TextDataSet.hpp"
//#include "Field.hpp"

template<typename DTYPE> class ParalleledCorpusDataset : public TextDataset<DTYPE>{
private:
  pair<string, string> m_languageName;

  map<int, string>* m_pIndex2Vocab;
  map<string, int>* m_pVocab2Frequency;
  map<string, int>* m_pVocab2Index;
  int n_vocabs;

  vector< pair<string, string> >* m_pairedSentences;
  vector< pair< vector<int>, vector<int> > >* m_pairedIndexedSentences;

  //-----넘겨주는 Data 관련!------//
  DTYPE **m_aaInput;
  DTYPE **m_aaLabel;

  int m_numOfInput;

  int m_dimOfInput;
  int m_dimOfLabel;

public:
  ParalleledCorpusDataset(string path, string srcName, string dstName, Field field);

  void                                               Alloc();

  void                                               MakeLineData();

  virtual void                                       BuildVocab();

  virtual                                            ~ParalleledCorpusDataset();

  void                                               Delete();


  vector< pair< string, string> >*        GetPairedSentences();

  vector< pair< vector<int>, vector<int> > >*                      GetmPairedIndexedSentences();
  //virtual std::vector<Tensor<DTYPE>*>*             GetData(int idx);

  void                                               MakeInputData();

  void                                               MakeLabelData();


  // void print() {
  //   int size = this->GetLineLength();
  //   size /= 2;
  //
  //   int n1, n2;
  //
  //   for(int i=0 ; i<size ; i++) {
  //     n1 = ((*m_pairedIndexedSentences)[i].first).size();
  //     n2 = ((*m_pairedIndexedSentences)[i].second).size();
  //
  //     cout<<"eng size : "<<n1<<endl;
  //     // for(int j=0 ; j<n1 ; j++)
  //     //   cout<<*(*m_pairedIndexedSentences)[i].first+j<<" ";
  //     // cout<<endl;
  //
  //     cout<<"fra size : "<<n2<<endl;
  //     // for(int j=0 ; j<n1 ; j++)
  //     //   cout<<*(*m_pairedIndexedSentences)[i].second+j<<" ";
  //     // cout<<endl;
  //   }
  //
  // }
};


template<typename DTYPE> ParalleledCorpusDataset<DTYPE>::ParalleledCorpusDataset(string path, string srcName, string dstName, Field field) : TextDataset<DTYPE>::TextDataset(path, field) {

  m_languageName = make_pair(srcName, dstName);

  m_pairedSentences = NULL;
  m_pairedIndexedSentences = NULL;

  m_aaInput = NULL;
  m_aaLabel = NULL;
  m_numOfInput = 0;
  m_dimOfInput = 0;
  m_dimOfLabel = 0;

  Alloc();
}

template<typename DTYPE> ParalleledCorpusDataset<DTYPE>::~ParalleledCorpusDataset() {
    cout << "ParalleledCorpusDataset 소멸자 호출" << endl;
    Delete();
}

template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::Delete() {
  if(m_pairedSentences) {
    delete m_pairedSentences;
    m_pairedSentences = NULL;
  }

  if(m_pairedIndexedSentences) {
    delete m_pairedIndexedSentences;
    m_pairedIndexedSentences = NULL;
  }
}


template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::Alloc() {

  m_pairedSentences = new vector< pair<string, string> >();
  m_pairedIndexedSentences = new vector< pair< vector<int>, vector<int> > >();

  m_pIndex2Vocab = this->GetpIndex2Vocab();
  m_pVocab2Frequency = this->GetpVocab2Frequency();
  m_pVocab2Index = this->GetpVocab2Index();
  n_vocabs = this->GetNumberofVocabs();

  BuildVocab();

  m_numOfInput = m_pairedIndexedSentences->size();


  
  //print();                                //DEBUG
}

template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::BuildVocab() {

  MakeLineData();
  cout<<"<<<<<<<<<<<<<<<<  BuildVocab 호출 >>>>>>>>>>>>>>>>>>>>"<<endl;
  //cout << m_pairedSentences->size() << endl;
  vector<string> temp_words;
  vector<int> temp_first_indexed_words;
  vector<int> temp_second_indexed_words;
  pair< vector<int>, vector<int> > temp_pair;

  for(int i=0; i<m_pairedSentences->size(); i++){
  //for(int i=0; i<100; i++){                                               //DEBUG
    temp_words = this->SplitBy(m_pairedSentences->at(i).first, ' ');

    for(string word: temp_words){
      // cout << word << " "<<m_pVocab2Index->at(word)<<endl;                    //DEBUG
      temp_first_indexed_words.push_back(m_pVocab2Index->at(word));
    }

     //cout<<endl<<"eng size : "<<temp_first_indexed_words.size()<<endl;             //DEBUG

////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    temp_words = this->SplitBy(m_pairedSentences->at(i).second, ' ');


    for(string word: temp_words){
      temp_second_indexed_words.push_back(m_pVocab2Index->at(word));
    }

    //cout<<endl<<"fra size : "<<temp_first_indexed_words.size()<<endl;             //DEBUG


    temp_pair = make_pair(temp_first_indexed_words, temp_second_indexed_words);

    m_pairedIndexedSentences->push_back(temp_pair);


    // cout<<endl<<"eng size : "<<m_pairedIndexedSentences->back().first.size()<<endl;             //DEBUG
    // cout<<endl<<"fra size : "<<m_pairedIndexedSentences->back().second.size()<<endl;             //DEBUG



    temp_first_indexed_words.clear();
    temp_second_indexed_words.clear();
  }
  m_pairedIndexedSentences->shrink_to_fit();
}

template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::MakeLineData() { // 확인완료

    cout<<"<<<<<<<<<<<<<<<<  MakeLineData  >>>>>>>>>>>>>>>>>>>>"<<endl;
    //cout<<strlen(TextData)<<endl;
    char* token = strtok(this->GetTextData(), "\t\n");
    char* last_sentence = NULL;

    while(token != NULL) {
      //cout<<token<<endl;              //DEBUG
      if(this->GetLineLength()%2==0){
        last_sentence = token;                                              // paired data를 만들기위해 앞에 오는 line 임시 저장
      }
      else {
        string str_last_sentence = this->Preprocess(last_sentence);
        string str_token = this->Preprocess(token);
        m_pairedSentences->push_back(make_pair(str_last_sentence, str_token));           // paired data 저장
        this->AddSentence(this->Preprocess(m_pairedSentences->back().first));
        this->AddSentence(this->Preprocess(m_pairedSentences->back().second));
      }
      //temp->line->push_back(token);                                         // 각 언어에 line 저장
      //MakeVocab(token);
      token = strtok(NULL, "\t\n");
      int temp_lineLength = this->GetLineLength();
      if(temp_lineLength%10000==0)
        cout<<"line_length = "<<temp_lineLength<<endl;

      this->SetLineLength(++temp_lineLength);
    }
    m_pairedSentences->shrink_to_fit();
    //text_lines /=2;
  }


template<typename DTYPE>  vector< pair<string, string> >* ParalleledCorpusDataset<DTYPE>::GetPairedSentences() {
  return m_pairedSentences;
}

template<typename DTYPE>  vector< pair< vector<int>, vector<int> > >* ParalleledCorpusDataset<DTYPE>::GetmPairedIndexedSentences() {
  return m_pairedIndexedSentences;
}

template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::MakeInputData() {
  int line_size = m_pairedIndexedSentences->size();


}

template<typename DTYPE> void ParalleledCorpusDataset<DTYPE>::MakeLabelData() {

}

#endif
