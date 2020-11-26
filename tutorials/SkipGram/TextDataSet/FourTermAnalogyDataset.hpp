#ifndef __ACCURACY_DATASET_HPP__
#define __ACCURACY_DATASET_HPP__    value

#include "../TextDataSet.hpp"
//#include "Field.hpp"

template<typename DTYPE> class FourTermAnalogyDataset : public TextDataset<DTYPE>{
private:
  DTYPE **m_aaInput;
  DTYPE **m_aaLabel;

  int m_numOfInput;             //input data의 개수!!!

  int m_dimOfInput;
  int m_dimOfLabel;

  vector< vector<string> >* wordTextData;

  map<int, string>* m_pIndex2Vocab;
  map<string, int>* m_pVocab2Frequency;
  map<string, int>* m_pVocab2Index;
  int n_vocabs;
  int n_words;

public:
  FourTermAnalogyDataset(string path, Field field);

  void                                   Alloc();

  virtual void                           BuildVocab();

  virtual                                ~FourTermAnalogyDataset();

  void                                   Delete();

  void                                   MakeInputData();
  void                                   MakeLabelData();

  virtual std::vector<Tensor<DTYPE>*>*   GetData(int idx);
  virtual int                            GetLength();

};


template<typename DTYPE> FourTermAnalogyDataset<DTYPE>::FourTermAnalogyDataset(string path, Field field) : TextDataset<DTYPE>::TextDataset(path, field) {
  m_aaInput = NULL;
  m_aaLabel = NULL;
  m_numOfInput = 0;
  m_dimOfInput = 0;
  m_dimOfLabel = 0;

  m_pIndex2Vocab = NULL;
  m_pVocab2Frequency = NULL;
  m_pVocab2Index = NULL;
  n_vocabs = 0;
  n_words = 0;

  wordTextData = NULL;
  Alloc();
}

template<typename DTYPE> void FourTermAnalogyDataset<DTYPE>::Alloc() {
  
  wordTextData = new vector<vector<string>>;

  BuildVocab();

  m_pIndex2Vocab = this->GetpIndex2Vocab();
  m_pVocab2Frequency = this->GetpVocab2Frequency();
  m_pVocab2Index = this->GetpVocab2Index();
  n_vocabs = this->GetNumberofVocabs();
  n_words = this->GetNumberofWords();

  m_numOfInput = n_words/4;     
  TextDataset<DTYPE>::SetInputDim(3);
  TextDataset<DTYPE>::SetLabelDim(1);
  m_dimOfInput = 3;               
  m_dimOfLabel = 1;      
  m_aaInput = new DTYPE *[m_numOfInput];
  m_aaLabel = new DTYPE *[m_numOfInput];

  MakeInputData();
}

template<typename DTYPE> void FourTermAnalogyDataset<DTYPE>::BuildVocab() {

    char* token = strtok(this->GetTextData(), "\n");
    char* last_sentence = NULL;

    while(token != NULL) {
        string temp = TextDataset<DTYPE>::Preprocess(token);
        TextDataset<DTYPE>::AddSentence(temp);
        vector<string> each_line = TextDataset<DTYPE>::SplitBy(temp, ' ');
        each_line.shrink_to_fit();
        wordTextData->push_back(each_line);
        token = strtok(NULL, "\n");
        int temp_lineLength = this->GetLineLength();
        if(temp_lineLength%10000==0)
        cout<<"line_length = "<<temp_lineLength<<endl;

        this->SetLineLength(++temp_lineLength);
    }
    cout << wordTextData->size() << endl;

    cout << "ACCURACY BuildVocab 완료" << endl;

}

template<typename DTYPE> void FourTermAnalogyDataset<DTYPE>::MakeInputData(){
    cout << "ACCURACY Make Input Data 호출" << endl;
    int textIndex = 0, inputIndex = 0, miss = 0;
    int w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    int flag = 0;
    for (int i = 0; i < m_numOfInput; i++) {
      vector<string> each_line = wordTextData->at(textIndex);

      for(int j=0; j<each_line.size(); j++){
        //cout << wordTextData->at(textIndex)->at(j) << " ";
      if(m_pVocab2Index->find(each_line.at(j))==m_pVocab2Index->end())
        flag = 1;
      }
      //cout << endl;
      if(flag){
        miss++;
        continue;
      }
      w0 = m_pVocab2Index->at(each_line.at(0));
      w1 = m_pVocab2Index->at(each_line.at(1));
      w2 = m_pVocab2Index->at(each_line.at(2));
      w3 = m_pVocab2Index->at(each_line.at(3));

      textIndex++;

        // //word2index에 없는 단어는 pass하기!
        // if(w0 == -1 || w1 == -1 || w2 == -1 || w3 == -1 ){
            
        // }
        m_aaInput[inputIndex] = new DTYPE[m_dimOfInput];
        m_aaLabel[inputIndex] = new DTYPE[m_dimOfLabel];
        m_aaInput[inputIndex][0] = w0;
        m_aaInput[inputIndex][1] = w1;
        m_aaInput[inputIndex][2] = w2;
        m_aaLabel[inputIndex][0] = w3;

        //std::cout<<m_pIndex2Vocab->at(w0)<<" "<<m_pIndex2Vocab->at(w1)<<" "<<m_pIndex2Vocab->at(w2)<<" "<<m_pIndex2Vocab->at(w3)<<'\n';

        inputIndex++;
    }

    m_numOfInput -= miss;

    std::cout<<"최종 m_numOfInput 개수 : "<<m_numOfInput<<"  "<<inputIndex<<'\n';
 
}

template<typename DTYPE> void FourTermAnalogyDataset<DTYPE>::MakeLabelData(){

      for (int i = 0; i < m_numOfInput; i++) {

          m_aaLabel[i] = new DTYPE[1];
          //positive sample
          m_aaLabel[i][0] = (DTYPE)0;       //positive sample은 항상 맨 처음
      }
}

template<typename DTYPE> void FourTermAnalogyDataset<DTYPE>::Delete() {

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

template<typename DTYPE> std::vector<Tensor<DTYPE> *>* FourTermAnalogyDataset<DTYPE>::GetData(int idx) {
      std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

      Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfInput);
      Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfLabel);

      for (int i = 0; i < m_dimOfInput; i++) {
          (*input)[i] = m_aaInput[idx][i];
      }

      for (int i = 0; i < m_dimOfLabel; i++) {
          (*label)[i] = m_aaLabel[idx][i];
      }

      result->push_back(input);
      result->push_back(label);

      return result;
}



template<typename DTYPE> FourTermAnalogyDataset<DTYPE>::~FourTermAnalogyDataset() {
    cout << "FourTermAnalogyDataset 소멸자 호출" << endl;
    Delete();
}

template<typename DTYPE> int FourTermAnalogyDataset<DTYPE>::GetLength() {
        return m_numOfInput;
}
#endif
