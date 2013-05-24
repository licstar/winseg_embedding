#include <string>
#include <map>
#include <vector>
#include <algorithm>
using namespace std;

#define MAX_STRING 1000

void readWord(char *word, FILE *fin){
	int a=0, ch;

	while (!feof(fin)) {
		ch=fgetc(fin);

		if (ch==13) continue;

		if ((ch==' ') || (ch=='\t') || (ch=='\n')) {
			if (a>0) {
				if (ch=='\n') ungetc(ch, fin);
				break;
			}

			if (ch=='\n') {
				strcpy(word, (char *)"</s>");
				return;
			}
			else continue;
		}

		word[a]=ch;
		a++;

		if (a>=MAX_STRING) {
			printf("Too long word found!\n");   //truncate too long words
			a--;
		}
	}
	word[a]=0;
}

map<string, int> chk;

data_t getWord(char *word){
	data_t ret;
	string w = word;
	map<string, int>::iterator it = chk.find(w);
	if(it != chk.end()){
		ret.word = it->second;
	}else{
		ret.word = chk["unknown"];
		printf("Error: unknown word!!\n");
	}
	return ret;
}

data_t readWordIndex(FILE *fin){
	char word[MAX_STRING];
	data_t ret;
	ret.word = 0;

	readWord(word, fin);
	if (feof(fin)) return ret;

	ret = getWord(word);

	return ret;
}

void init(const char *dict_file){
	/*chk["</s>"] = 0;
	chk["unknown"] = 1;
	chk["padding"] = 2;
	chk["number"] = 3;
	chk["letter"] = 4;
	chk["numletter"] = 5;
	*/
	char ch[100];
	FILE *fin = fopen(dict_file, "r");
	while(fgets(ch, sizeof(ch), fin)){
		int len = strlen(ch);
		while((ch[len-1] == '\r' || ch[len-1] == '\n') && len > 0){
			ch[len-1] = 0;
			len--;
		}
		len = chk.size();
		chk[ch] = len;
	}
	fclose(fin);

	for(map<string, int>::iterator it = chk.begin(); it != chk.end(); it++){
		//printf("%s", it->first.c_str());
	}
}


bool operator <(const data_t &a, const data_t &b){
	return a.word < b.word;
}

bool operator ==(const data_t &a, const data_t &b){
	return a.word == b.word;
}

void addLineData(vector<vector<data_t> > &sortData, vector<data_t> &vec, int *specialCount){
	data_t padding; //这个想办法初始化一下
	padding.word = 2;
	
	int hw = (window_size-1)/2;

	for(int j = 0; j < (int)vec.size(); j++){
		vector<data_t> line(window_size);
		for(int k = hw; k > 0; k--){
			if(j-k >= 0){
				line[hw - k] = vec[j-k];
			}else{
				line[hw - k] = padding; //PADDING
			}
		}
		for(int k = 1; k <= hw; k++){
			if(j+k < (int)vec.size()){
				line[hw + k] = vec[j+k];
			}else{
				line[hw + k] = padding; //PADDING
			}
		}
		line[hw] = vec[j];
		if(vec[j].word < 6)
			specialCount[vec[j].word]++;
		sortData.push_back(line);
	}
}

void readAllData(const char *file, int window_size, data_t *&data, int &N){
	//vector<vector<data_t > > mydata;
	FILE *fi=fopen(file, "rb");

	vector<data_t> line;

	vector<vector<data_t> > sortData;
	int specialCount[6]={0};

	//读取数据
	N = 0;
	while(1){
		data_t dt = readWordIndex(fi);
		if (feof(fi)) break;
		line.push_back(dt);

		if(dt.word == 0){
			line.pop_back();
			//mydata.push_back(line);
			addLineData(sortData, line, specialCount);

			N += line.size();
			line.clear();
		}
	}
	fclose(fi);

	printf("data: N(words):%d, ", N);
	printf("unknown:%d number:%d letter:%d numletter:%d\n", specialCount[1], specialCount[3], specialCount[4], specialCount[5]);

	//去重
	sort(sortData.begin(), sortData.end());
	sortData.erase(unique(sortData.begin(), sortData.end()), sortData.end());

	N = sortData.size();
	printf("unique data:   %d\n", N);

	//乱序
	for(int i = 0; i < N; i++){
		swap(sortData[i], sortData[rand()%N]);
	}

	data = new data_t[N * window_size];

	for(size_t i = 0; i < sortData.size(); i++){
		vector<data_t> &vec = sortData[i];
		for(size_t k = 0; k < vec.size(); k++){
			data[i*window_size + k] = vec[k];
		}
	}

}