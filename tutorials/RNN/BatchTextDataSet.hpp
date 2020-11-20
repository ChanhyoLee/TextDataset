#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string.h>

#include "../../WICWIU_src/Tensor.hpp"
//#include "../../WICWIU_src/DataLoader.hpp"        // 왜 추가한거지?   Dataset때문에 추가한거 같음... 추측임

using namespace std;

enum OPTION {
    ONEHOT,
    CBOW
};


void MakeOneHotVector(int* onehotvector, int vocab_size, int index){

    for(int i=0; i<vocab_size; i++){
        if(i==index)
            onehotvector[i] = 1;
        else
            onehotvector[i] = 0;
    }
}

//: public Dataset<DTYPE>{

template<typename DTYPE>
class BatchTextDataSet {
private:

    char* vocab ;
    char* TextData;

    int vocab_size;
    int text_length;

    Tensor<DTYPE>* input;
    Tensor<DTYPE>* label;

    OPTION option;

    //batch를 위해 추가
    int batchsize;
    int timesize;

    int VOCAB_LENGTH;

public:
    BatchTextDataSet(string File_Path, int batch_size, int vocab_length, OPTION pOption) {
        vocab = NULL;
        TextData = NULL;

        vocab_size = 0;
        text_length = 0;

        input = NULL;
        label = NULL;

        option = pOption;

        timesize = 0;
        batchsize = batch_size;
        VOCAB_LENGTH = vocab_length;

        Alloc(File_Path);
    }

    virtual ~BatchTextDataSet() {
        Delete();
    }

    //왜 굳이 virtual인거지?
    void                                  Alloc(string File_Path);

    void                                  Delete();

    void                                  FileReader(string pFile_Path);
    void                                  MakeVocab();

    void                                  MakeInputData();
    void                                  MakeLabelData();

    int                                   char2index(char c);

    char                                  index2char(int index);

    Tensor<DTYPE>*                        GetInputData();

    Tensor<DTYPE>*                        GetLabelData();

    int                                   GetTextLength();

    int                                   GetVocabLength();

    int                                   GetTimeSize();

    //virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    //virtual int                           GetLength();

};

template<typename DTYPE> void BatchTextDataSet<DTYPE>::Alloc(string File_Path) {

    vocab = new char[VOCAB_LENGTH];
    //File_Reader
    FileReader(File_Path);

    //make_vocab
    MakeVocab();

    //make_Input_data
    MakeInputData();

    //make_label_data
    MakeLabelData();
}


template<typename DTYPE> void BatchTextDataSet<DTYPE>::Delete() {
    delete []vocab;
    delete []TextData;
}

template<typename DTYPE> void BatchTextDataSet<DTYPE>::FileReader(string pFile_Path) {
    ifstream fin;
    fin.open(pFile_Path);

    if(fin.is_open()){

      //파일 사이즈 구하기
      fin.seekg(0, ios::end);
      text_length = fin.tellg();
      fin.seekg(0, ios::beg);        //포인터를 다시 시작위치로 바꿈

      //파일 길이만큼 할당
      TextData = new char[text_length];

      //파일 읽기
      fin.read(TextData, text_length);
      //여기에 NULL 추가

    //  std::cout<<TextData<<'\n';
      //소문자로 변환
        for(int i=0; i<text_length; i++)
            TextData[i] = tolower(TextData[i]);
    //  std::cout<<TextData<<'\n';


    }
    fin.close();

    timesize = text_length/batchsize;
}

template<typename DTYPE> void BatchTextDataSet<DTYPE>::MakeVocab(){

    int flag = 0;
    for(int i=0; i<text_length; i++){

        flag = 0;
        vocab_size = (int)strlen(vocab);

        for(int j=0; j<vocab_size; j++){
            if(vocab[j]==TextData[i])
              flag = 1;
            }

        if(flag==0){
          vocab[vocab_size] = TextData[i];
        }
    }

    vocab_size = (int)strlen(vocab)+1;
    //for(int i=0; i<vocab_size; i++)
    //    std::cout<<i<<"번째 vocab :"<<int(vocab[i])<<'\n';
    sort(vocab, vocab+vocab_size-1);


}

template<typename DTYPE> void BatchTextDataSet<DTYPE>::MakeInputData(){


    if(option == ONEHOT){
        int* onehotvector = new int[vocab_size];

        input = new Tensor<DTYPE>(timesize, batchsize, 1, 1, vocab_size);

        for(int ti=0; ti<timesize; ti++){
            for(int ba = 0; ba<batchsize; ba++){
                MakeOneHotVector(onehotvector, vocab_size, char2index(TextData[ti*batchsize + ba]));
                for(int col=0; col<vocab_size; col++){
                    (*input)[Index5D(input->GetShape(), ti, ba, 0, 0, col)] = onehotvector[col];
                }
            }
        }
    }

}

template<typename DTYPE> void BatchTextDataSet<DTYPE>::MakeLabelData(){

    if(option == ONEHOT){
        int* onehotvector = new int[vocab_size];

        label = new Tensor<float>(timesize, batchsize, 1, 1, vocab_size);

        for(int ti=0; ti<timesize; ti++){
            for(int ba=0; ba<batchsize; ba++){

                //마지막 data
                if((ti*batchsize+ba) == text_length-1){
                    MakeOneHotVector(onehotvector, vocab_size, vocab_size-1);
                    for(int j=0; j<vocab_size; j++){
                        (*label)[Index5D(label->GetShape(), ti, ba, 0, 0, j)] = onehotvector[j];
                  }
                  continue;
                }

                MakeOneHotVector(onehotvector, vocab_size, char2index(TextData[ti*batchsize + ba +1]));
                for(int j=0; j<vocab_size; j++){
                    (*label)[Index5D(label->GetShape(), ti, ba, 0, 0, j)] = onehotvector[j];
                }
            }
        }
    }
}

template<typename DTYPE> int BatchTextDataSet<DTYPE>::char2index(char c){

    for(int index=0; index<vocab_size; index++){
        if(vocab[index]==c)
          return index;
    }
    return -1;
}

template<typename DTYPE> char BatchTextDataSet<DTYPE>::index2char(int index){

    return vocab[index];
}

template<typename DTYPE> Tensor<DTYPE>* BatchTextDataSet<DTYPE>::GetInputData(){

    return input;
}

template<typename DTYPE> Tensor<DTYPE>* BatchTextDataSet<DTYPE>::GetLabelData(){
    return label;
}

template<typename DTYPE> int BatchTextDataSet<DTYPE>::GetTextLength(){
    return text_length;
}

template<typename DTYPE> int BatchTextDataSet<DTYPE>::GetVocabLength(){
    return vocab_size;
}

template<typename DTYPE> int BatchTextDataSet<DTYPE>::GetTimeSize(){
    return timesize;
}
