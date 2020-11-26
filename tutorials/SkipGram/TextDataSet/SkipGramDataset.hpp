#ifndef __SKIPGRAM_DATASET_HPP__
#define __SKIPGRAM_DATASET_HPP__

#include "../TextDataSet.hpp"
//#include "Field.hpp"

template<typename DTYPE> class SkipGramDataset : public TextDataset<DTYPE>{
private:
  DTYPE **m_aaInput;
  DTYPE **m_aaLabel;

  int m_numOfInput;             
  int m_window;                 
  int m_negative;

  map<int, string>* m_pIndex2Vocab;
  map<string, int>* m_pVocab2Frequency;
  map<string, int>* m_pVocab2Index;
  int n_vocabs;
  int n_words;

  vector<string>* wordTextData;

public:
  SkipGramDataset(string path, int window, int negative, Field field);

  void                                   Alloc(int window, int negative);

  virtual void                           BuildVocab();

  virtual                                ~SkipGramDataset();

  void                                   Delete();

  void                                   MakeInputData();
  void                                   MakeLabelData();

  virtual std::vector<Tensor<DTYPE>*>*   GetData(int idx);
  virtual int                            GetLength();


};


template<typename DTYPE> SkipGramDataset<DTYPE>::SkipGramDataset(string path, int window, int negative, Field field) : TextDataset<DTYPE>::TextDataset(path, field) {
  m_aaInput = NULL;
  m_aaLabel = NULL;

  m_numOfInput = 0;
  m_window     = 0;
  m_negative   = 0;

  TextDataset<DTYPE>::SetInputDim(0);
  TextDataset<DTYPE>::SetLabelDim(0);

  m_pIndex2Vocab = NULL;
  m_pVocab2Frequency = NULL;
  m_pVocab2Index = NULL;
  n_vocabs = 0;
  n_words = 0;

  wordTextData = NULL;
  Alloc(window, negative);
}


template<typename DTYPE> void SkipGramDataset<DTYPE>::Alloc(int window, int negative) {
  wordTextData = new vector<string>();
  wordTextData->push_back("<s>");
  wordTextData->push_back("<e>");

  BuildVocab();

  m_pIndex2Vocab = this->GetpIndex2Vocab();
  m_pVocab2Frequency = this->GetpVocab2Frequency();
  m_pVocab2Index = this->GetpVocab2Index();
  n_vocabs = this->GetNumberofVocabs();
  n_words = this->GetNumberofWords();

  m_window     = window;
  m_negative   = negative;

  m_numOfInput = n_words * (m_window - 1);
  TextDataset<DTYPE>::SetInputDim(m_negative+2);
  TextDataset<DTYPE>::SetLabelDim(m_negative+1);

  m_aaInput = new DTYPE *[m_numOfInput];
  m_aaLabel = new DTYPE *[m_numOfInput];

  MakeInputData();
}

template<typename DTYPE> void SkipGramDataset<DTYPE>::BuildVocab() {
  cout << "SKIPGRAM Build Vocab 호출" << endl;
  string str = TextDataset<DTYPE>::Preprocess(this->GetTextData());
  vector<string> temp_vect = TextDataset<DTYPE>::SplitBy(str, ' ');
  for(int i=0; i<temp_vect.size(); i++){
    wordTextData->push_back(temp_vect.at(i));
    TextDataset<DTYPE>::AddWord(temp_vect.at(i));
  }
  //TextDataset<DTYPE>::AddSentence(str);
}

template<typename DTYPE> void SkipGramDataset<DTYPE>::MakeInputData(){
  cout << "SKIPGRAM MakeInputData 호출" << endl;
  //subsampling때문에 추가
  unsigned long long next_random = 1;       // 출력은 %lld임!
  float sample = 1e-4;
  //여기까지 subsamling때문에 추가


  //이게 시간이 오래걸리지 않을까?....
  std::cout<<"------------------------------MakeskipgramInputdata---------------------"<<'\n';

  srand(time(NULL));                      //이 함수는 main에서 한번만 호출하면 됨!!!!   이거 그래서 수정 필요!!!
  int offsetIndex = 0;
  int index = 0;

  //context에 해당하는 index에 접근하기 위한 offset
  int contextOffset[m_window-1];
  for(int i=0; i<m_window/2; i++){
      contextOffset[index++] = i+1;
      contextOffset[index++] = -(i+1);
  }

  int centerIndex     = 0;
  int nonContextIndex = 0;

  for (int i = 0; i < m_numOfInput; i++) {
    m_aaInput[i] = new DTYPE[TextDataset<DTYPE>::GetInputDim()];

    //center word
    if(i%1000000==0){
      cout << i << "/" << m_numOfInput << endl;
    }
    m_aaInput[i][0] = (DTYPE)m_pVocab2Index->at(wordTextData->at(centerIndex));


    if(centerIndex+contextOffset[offsetIndex] < 0){                   // vocab index가 0보다 작으면 안되기 때문.. 그럴경우 <sos>로 저장
        m_aaInput[i][1] = (DTYPE)m_pVocab2Index->at("<s>");                   // int를 float로 강제 형변환
    } else if(centerIndex+contextOffset[offsetIndex] >= n_words){    // 왜 word_num? 중복 포함하면 안되지않나?
        m_aaInput[i][1] = (DTYPE)m_pVocab2Index->at("<e>");                   // 아무튼 vocab index보다 크면 <eos>로 저장
    } else{
        // cout << "index: " << centerIndex+contextOffset[offsetIndex] << "/" << wordTextData->size() << endl;
        // cout << "word: " << wordTextData->at(centerIndex+contextOffset[offsetIndex]) << endl;
        m_aaInput[i][1] = (DTYPE)m_pVocab2Index->at(wordTextData->at(centerIndex+contextOffset[offsetIndex]));      // 그냥 맞는 인덱스값 저장
    }

    for (int d = 2; d < TextDataset<DTYPE>::GetInputDim(); d++) {
          nonContextIndex = rand()%n_words;

          if(nonContextIndex == centerIndex){
                d--;
                continue;
          }
          // cout << "nonContextIndex: " << nonContextIndex << endl;
          // cout << "word: " << wordTextData->at(nonContextIndex) << endl;

          m_aaInput[i][d] = (DTYPE)m_pVocab2Index->at(wordTextData->at(nonContextIndex));
          for(int j=0; j<m_window-1; j++){
              if(nonContextIndex==centerIndex+contextOffset[j]){
                  d--;
              }
          }
    }

    offsetIndex++;
    if(offsetIndex == m_window-1){
      centerIndex++;
      offsetIndex = 0;
    }
  }
}

template<typename DTYPE> void SkipGramDataset<DTYPE>::MakeLabelData(){

      for (int i = 0; i < m_numOfInput; i++) {

          m_aaLabel[i] = new DTYPE[1];
          //positive sample
          m_aaLabel[i][0] = (DTYPE)0;       //positive sample은 항상 맨 처음
      }
}

template<typename DTYPE> std::vector<Tensor<DTYPE> *>* SkipGramDataset<DTYPE>::GetData(int idx) {
      std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

      Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(1, 1, 1, 1, TextDataset<DTYPE>::GetInputDim());
      Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, TextDataset<DTYPE>::GetLabelDim());

      for (int i = 0; i < TextDataset<DTYPE>::GetInputDim(); i++) {
          //이거는 전체 단어의 개수 안 맞춰주면 이렇게 됨!!!
          if(m_aaInput[idx][i]==-1)
              std::cout<<'\n'<<"****************************************************************************************음수존재..."<<'\n';
          (*input)[i] = m_aaInput[idx][i];
      }

      //(*label)[ (int)m_aaLabel[idx][0] ] = 1.f;
      (*label)[0] = 1.f;

      result->push_back(input);
      result->push_back(label);

      return result;
}


template<typename DTYPE> void SkipGramDataset<DTYPE>::Delete() {

   if(m_aaInput) {
    delete[] m_aaInput;
    m_aaInput = NULL;
  }

  if(m_aaLabel) {
    delete[] m_aaLabel;
    m_aaLabel = NULL;
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
}

template<typename DTYPE> SkipGramDataset<DTYPE>::~SkipGramDataset() {
    cout << "SkipGramDataset 소멸자 호출" << endl;
    Delete();
}

template<typename DTYPE> int SkipGramDataset<DTYPE>::GetLength() {
        return m_numOfInput;
}


#endif
