#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string>
#include <string.h>
#include <vector>

using namespace std;

class Langs {

private:
  string lang1;               // Language 1 이름
  string lang2;               // Language 2 이름
  vector<string> line1;       // Language 1 data
  vector<string> line2;       // Language 2 data
  vector<string> vocab1;    // Language 1 Vocab
  vector<string> vocab2;    // Language 2 Vocab

  char* TextData;           // 전체 text file 받을 char 배열 포인터

  vector<string> wordTextData1;       // line1의 전체 vocab
  vector<string> wordTextData2;       // line2의 전체 vocab

  vector<int> wordFrequency1;         // vocab1 frequency
  vector<int> wordFrequency2;         // vocab2 frequency

  int text_lines;                     // text file의 전체 line수

  int vocab_size1;                    // vocab1의 수
  int vocab_size2;                    // vocab2의 수

  int text_length;        // 전체 text의 길이
  int word_num;           // lang1 + lang2 전체 vocab 수

public:
  Langs(string File_Path, string plang1, string plang2) {

    lang1 = plang1;
    lang2 = plang2;

    vocab_size1 = 0;
    vocab_size2 = 0;

    text_lines = 0;
    word_num = 0;

    Alloc(File_Path);
  }

  void Delete()
  {
    delete[] TextData;
  }

  ~Langs()
  {
    Delete();
  }

  void Alloc(string File_Path) {

    FileReader(File_Path);

    Seperate_Langs();

    line1.shrink_to_fit();
    line2.shrink_to_fit();

    MakeVocab(&line1, &vocab1, &wordTextData1, &wordFrequency1);
    MakeVocab(&line2, &vocab2, &wordTextData2, &wordFrequency2);

    vocab_size1 = vocab1.size();
    vocab_size2 = vocab2.size();
    word_num = wordTextData1.size() + wordTextData2.size();

    vocab1.shrink_to_fit();
    vocab2.shrink_to_fit();
    wordTextData1.shrink_to_fit();
    wordTextData2.shrink_to_fit();
    wordFrequency1.shrink_to_fit();
    wordFrequency2.shrink_to_fit();
  }

  void FileReader(string pFile_Path) {

    cout<<"<<<<<<<<<<<<<<<<  FileReader  >>>>>>>>>>>>>>>>>>>>"<<endl;

    ifstream fin;
    fin.open(pFile_Path);

    if(fin.is_open()) {

      fin.seekg(0, ios::end);
      text_length = fin.tellg();
      fin.tellg();
      fin.seekg(0, ios::beg);

      TextData = new char[text_length];

      //파일 읽기
      fin.read(TextData, text_length);


    text_length = strlen(TextData);     //strlen원리가 NULL를 찾을 때 까지여서 마지막에 NULL이 자동으로 추가된거 같음!

    fin.close();
    }
  }

  void Seperate_Langs() {

    cout<<"<<<<<<<<<<<<<<<<  Seperate_Langs  >>>>>>>>>>>>>>>>>>>>"<<endl;

    char* token = strtok(TextData, "\t\n");

    while(token != NULL) {
      if(text_lines%2==0)
        line1.push_back(token);
      else
        line2.push_back(token);

      token = strtok(NULL, "\t\n");
      ++text_lines;

      if(text_lines%10000==0)
        cout<<"text_lines = "<<text_lines<<endl;
    }
  }

  void MakeVocab(vector<string>* line, vector<string>* vocab, vector<string>* wordTextData, vector<int>* wordFrequency) {

    cout<<"<<<<<<<<<<<<<<<<  MakeVocab  >>>>>>>>>>>>>>>>>>>>"<<endl;

    int flag = 0;
    char* token = NULL;
    int word_count =0;
    string stemp;
    char* temp;                               // 한번 더 생각해보기


    for(int i=0 ; i<text_lines/2 ; i++) {

      stemp = Remove((*line)[i], ".?!");         // remove char 전처리 가능 구간

      temp = new char[stemp.length()+1];
      strcpy(temp,stemp.c_str());

      token = strtok(temp," ");

      while(token != NULL) {
        wordTextData->push_back(token);

        for(int i = 0; i<vocab->size() ; i++) {
          if(strcmp((*vocab)[i].c_str(),token)==0) {
            flag = 1;
            (*wordFrequency)[i] += 1;
            break;
          }
        }

        if(flag == 0) {
          vocab->push_back(token);
          wordFrequency->push_back(1);
        }

        token = strtok(NULL," ");

        flag = 0;
      }

      free(temp);

      if(i%10000 == 0)
        cout<<"<<<<<<<<<<<<<<<< Vocab Text Lines = "<<i<<" >>>>>>>>>>>>>>>>>>>>>>>"<<endl;
    }
  }


  string Remove(string line, char* r) {

    int r_size = strlen(r);
    int flag = 0;

    for(int i=0 ; i<line.length(); i++) {
      for(int j=0 ; j<r_size ; j++) {
        if(line[i]==r[j]) {
          flag = 1;
          break;
        }
      }
      if(flag == 1)
        line.erase(i);

      flag = 0;
    }

    return line;
  }

  string GetLang1() {
    return lang1;
  }

  string GetLang2() {
    return lang2;
  }

  vector<string> GetLangLine1() {
    return line1;
  }

  vector<string> GetLangLine2() {
    return line2;
  }
  vector<string> GetLangVocab1() {
    return vocab1;
  }

  vector<string> GetLangVocab2() {
    return vocab2;
  }

  int GetLangVocabSize1() {
    return vocab_size1;
  }

  int GetLangVocabSize2() {
    return vocab_size2;
  }

  int GetLangWordSize1() {
    return wordTextData1.size();
  }

  int GetLangWordSize2() {
    return wordTextData2.size();
  }

  int GetLangTextLines() {
    return text_lines;
  }

  int GetWordNumber() {
    return word_num;
  }

  int GetTextLength() {
    return text_length;
  }
};

/*
int main()
{
  Langs lang("eng-fra.txt", "eng", "fra");

  vector<string> v1 = lang.GetLangVocab1();
  vector<string> v2 = lang.GetLangVocab2();
  vector<string> l1 = lang.GetLangLine1();
  vector<string> l2 = lang.GetLangLine2();

  cout<<lang.GetLang1()<<endl;
  cout<<lang.GetLang2()<<endl;
  cout<<"Vocab1 size = "<<lang.GetLangVocabSize1()<<endl;
  cout<<"Vocab2 size = "<<lang.GetLangVocabSize2()<<endl;
  cout<<"Word1 size = "<<lang.GetLangWordSize1()<<endl;
  cout<<"Word2 size = "<<lang.GetLangWordSize2()<<endl;
  cout<<"Word number = "<<lang.GetWordNumber()<<endl;
  cout<<"Text lines = " <<lang.GetLangTextLines()<<endl;
  cout<<"Text Length = "<<lang.GetTextLength()<<endl;


  for(int i=0 ; i<100 ; i++)
    cout<<l2[i]<<endl;

  return 0;
}
*/
