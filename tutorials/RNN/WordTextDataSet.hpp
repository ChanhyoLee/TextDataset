#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string.h>

#include "../../WICWIU_src/Tensor.hpp"

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

string replaceAll(const string &str, const string &pattern, const string &replace){

    string result = str;
    string::size_type pos = 0;
    string::size_type offset = 0;

    while( (pos = result.find(pattern, offset)) != string::npos){

        result.replace(result.begin() + pos, result.begin() + pos + pattern.size(), replace);
        offset = pos + replace.size();
    }

    return result;
}


template<typename DTYPE>
class WordTextDataSet {
private:

    char* vocab ;
    string *wordVocab;    //string 배열?


    char* TextData;
    string strTextData;

    int vocab_size;
    int text_length;

    Tensor<DTYPE>* input;
    Tensor<DTYPE>* label;

    OPTION option;

    int VOCAB_LENGTH;

public:
    WordTextDataSet(string File_Path, int vocab_length, OPTION pOption) {
        vocab = NULL;
        TextData = NULL;

        vocab_size = 0;
        text_length = 0;

        input = NULL;
        label = NULL;

        option = pOption;

        VOCAB_LENGTH = vocab_length;

        Alloc(File_Path);
    }

    virtual ~WordTextDataSet() {
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

    //virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    //virtual int                           GetLength();

};

template<typename DTYPE> void WordTextDataSet<DTYPE>::Alloc(string File_Path) {

    vocab = new char[VOCAB_LENGTH];
    wordVocab = new string[VOCAB_LENGTH];
    //File_Reader
    FileReader(File_Path);

    //make_vocab
    MakeVocab();

    //make_Input_data
    MakeInputData();

    //make_label_data
    MakeLabelData();
}


template<typename DTYPE> void WordTextDataSet<DTYPE>::Delete() {
    delete []vocab;
    delete []TextData;
}

template<typename DTYPE> void WordTextDataSet<DTYPE>::FileReader(string pFile_Path) {
    ifstream fin;
    fin.open(pFile_Path);


    if(fin.is_open()){

      //파일 사이즈 구하기
      fin.seekg(0, ios::end);
      text_length = fin.tellg();
      fin.seekg(0, ios::beg);        //포인터를 다시 시작위치로 바꿈

      //파일 길이만큼 할당
      strTextData.resize(text_length);

      //파일 읽기
      fin.read(&strTextData[0], text_length);

    }
    fin.close();

    //2가지 방법다 가능!!! - erase함수로 제거해도 좋고!, replaceAll로 제거하는것도 가능!
        strTextData.erase(std::remove(strTextData.begin(), strTextData.end(), char(0xa)), strTextData.end());   //lf : 다음 줄
        //strTextData.erase(std::remove(strTextData.begin(), strTextData.end(), char(0xd)), strTextData.end());   //cr : 커서를 앞으로 이동

        //CR은 space로 대체!
        //strTextData = replaceAll(strTextData, "\n", "");
        strTextData = replaceAll(strTextData, "\r", " ");

        //이걸 어떻게 처리해야될까....????
        strTextData = replaceAll(strTextData, "  ", " ");
        //strTextData = replaceAll(strTextData, "   ", " ");

    //특수문자도 없애면 좋나??
    strTextData.erase(std::remove(strTextData.begin(), strTextData.end(), ':'), strTextData.end());
    strTextData.erase(std::remove(strTextData.begin(), strTextData.end(), ','), strTextData.end());
    strTextData.erase(std::remove(strTextData.begin(), strTextData.end(), '.'), strTextData.end());

    // TextData = strTextData.c_str();

    //소문자로 변경하기
    transform(strTextData.begin(), strTextData.end(), strTextData.begin(), [](unsigned char c){return std::tolower(c);});

    std::cout<<strTextData<<'\n';

    //TextData = strTextData.c_str();

}

template<typename DTYPE> void WordTextDataSet<DTYPE>::MakeVocab(){

    int flag = 0;
    string token;
    int count=0;

    stringstream ss(strTextData);

    while(getline(ss, token, ' ')){

        flag = 0;

        vocab_size = count;

        //단어 중복 확인하기
        for(int i=0; i<vocab_size; i++){
            if(wordVocab[i] == token){
                flag = 1;
            }
        }

        //새로운 단어인 경우
        if(flag == 0){
          wordVocab[count] = token;
          count++;
        }
    }


    // vocab_size = (sizeof(wordVocab) / sizeof(string)) +1;
    vocab_size = count + 1;

    std::cout<<"vocab_size = "<<vocab_size<<'\n';

    for(int i=0; i<vocab_size; i++)
        std::cout<<i<<"번째 wordVocab = "<<wordVocab[i]<<'\n';

    sort(wordVocab, wordVocab+vocab_size-1);

    for(int i=0; i<vocab_size; i++)
        std::cout<<i<<"번째 wordVocab = "<<wordVocab[i]<<"---"<<'\n';


      // const char* what = wordVocab[11].c_str();
      //
      // std::cout<<int(what)<<'\n';

}

template<typename DTYPE> void WordTextDataSet<DTYPE>::MakeInputData(){

    if(option == ONEHOT){
        int* onehotvector = new int[vocab_size];

        input = new Tensor<DTYPE>(text_length, 1, 1, 1, vocab_size);

        for(int i=0; i<text_length; i++){
            MakeOneHotVector(onehotvector, vocab_size, char2index(TextData[i]));
            for(int j=0; j<vocab_size; j++){
                (*input)[Index5D(input->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
            }
        }
    }

}

template<typename DTYPE> void WordTextDataSet<DTYPE>::MakeLabelData(){

    if(option == ONEHOT){
        int* onehotvector = new int[vocab_size];

        label = new Tensor<float>(text_length, 1, 1, 1, vocab_size);

        for(int i=0; i<text_length; i++){

            //마지막 data
            if(i==text_length-1){
                MakeOneHotVector(onehotvector, vocab_size, vocab_size-1);
                for(int j=0; j<vocab_size; j++){
                    (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
              }
              continue;
            }

            MakeOneHotVector(onehotvector, vocab_size, char2index(TextData[i+1]));
            for(int j=0; j<vocab_size; j++){
                (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
            }
        }
    }
}

template<typename DTYPE> int WordTextDataSet<DTYPE>::char2index(char c){

    for(int index=0; index<vocab_size; index++){
        if(vocab[index]==c)
          return index;
    }
    return -1;
}

template<typename DTYPE> char WordTextDataSet<DTYPE>::index2char(int index){

    return vocab[index];
}

template<typename DTYPE> Tensor<DTYPE>* WordTextDataSet<DTYPE>::GetInputData(){

    return input;
}

template<typename DTYPE> Tensor<DTYPE>* WordTextDataSet<DTYPE>::GetLabelData(){
    return label;
}

template<typename DTYPE> int WordTextDataSet<DTYPE>::GetTextLength(){
    return text_length;
}

template<typename DTYPE> int WordTextDataSet<DTYPE>::GetVocabLength(){
    return vocab_size;
}
