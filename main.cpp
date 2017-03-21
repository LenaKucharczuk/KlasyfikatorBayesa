#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "Classifier.h"
#include "Windows.h"
#include <ctime>
//#include <boost/lexical_cast.hpp>

using namespace std;
void loadFromFile(Classifier& classifier, string filename);

int main()
{
	int par, seq;
	Classifier classifier, classifier2, classifier3, classifier4;
	string answer;
	vector<string> query;
	std::clock_t c_start, c_end;


	loadFromFile(classifier, "plik.txt");

	cout << "________________KLASYFIKATOR 1________________" << endl;
	classifier.train();

	query = { "2", "11", "21", "32", "45", "56", "65", "76", "86", "95", "108", "111", "127", "130", "143", "156", "166", "176", "187",
		"196", "209", "219", "224", "236", "242", "254", "266", "278", "282", "297", "309", "315", "322", "337", "348", "357", "367",
		"378", "380", "393", "405", "415", "426", "438", "448", "452", "468", "470", "485", "493" };
	c_start = std::clock();
	answer = classifier.ask(query);
	c_end = std::clock();
	if (!answer.empty())
		cout << "Odpowiedz: " << answer << ". ";
	else
		cout << "Twoja odpowiedz nie zostala znaleziona." << endl;
	par = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "GPU: Czas przetwarzania: " << par << " ms" << endl;

	c_start = std::clock();
	answer = classifier.sequentialAsk(query);
	c_end = std::clock();
	cout << "Odpowiedz: " << answer << ". ";
	seq = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "CPU: Czas przetwarzania: " << seq << " ms" << endl;
	cout << "Przetwarzanie na GPU trwalo o " << par - seq << " ms dluzej." << endl << endl;

	query = { "7", "19", "28", "30", "48", "56", "66", "72", "81", "93", "100", "119", "128", "130", "147", "155", "164", "171", "185",
		"192", "209", "219", "226", "230", "243", "253", "263", "278", "280", "291", "304", "311", "329", "339", "343", "354", "363",
		"374", "387", "399", "404", "410", "429", "437", "444", "457", "465", "478", "482", "499" };
	c_start = std::clock();
	answer = classifier.ask(query);
	c_end = std::clock();
	if (!answer.empty())
		cout << "Odpowiedz: " << answer << ". ";
	else
		cout << "Twoja odpowiedz nie zostala znaleziona." << endl;
	par = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "GPU: Czas przetwarzania: " << par << " ms" << endl;

	c_start = std::clock();
	answer = classifier.sequentialAsk(query);
	c_end = std::clock();
	cout << "Odpowiedz: " << answer << ". ";
	seq = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "CPU: Czas przetwarzania: " << seq << " ms" << endl;
	cout << "Przetwarzanie na GPU trwalo o " << par - seq << " ms dluzej." << endl << endl;

	//////////////////////////////////////////////////////////////////

	loadFromFile(classifier2, "plik2.txt");
	cout << "________________KLASYFIKATOR 2________________" << endl;
	classifier2.train();

	query = { "rain", "mild", "high", "strong" };
	c_start = std::clock();
	answer = classifier2.ask(query);
	c_end = std::clock();
	if (!answer.empty())
		cout << "Odpowiedz: " << answer << ". ";
	else
		cout << "Twoja odpowiedz nie zostala znaleziona." << endl;
	par = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "GPU: Czas przetwarzania: " << par << " ms" << endl;

	c_start = std::clock();
	answer = classifier2.sequentialAsk(query);
	c_end = std::clock();
	cout << "Odpowiedz: " << answer << ". ";
	seq = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "CPU: Czas przetwarzania: " << seq << " ms" << endl;
	cout << "Przetwarzanie na GPU trwalo o " << par - seq << " ms dluzej." << endl << endl;

	/////////////////////////////////////////////////////////////

	loadFromFile(classifier3, "plik3.txt");
	cout << "________________KLASYFIKATOR 3________________" << endl;
	classifier3.train();

	query = { "5", "10", "22", "38", "47", "56", "63", "77", "84", "94" };
	c_start = std::clock();
	answer = classifier3.ask(query);
	c_end = std::clock();
	if (!answer.empty())
		cout << "Odpowiedz: " << answer << ". ";
	else
		cout << "Twoja odpowiedz nie zostala znaleziona." << endl;
	par = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "GPU: Czas przetwarzania: " << par << " ms" << endl;

	c_start = std::clock();
	answer = classifier3.sequentialAsk(query);
	c_end = std::clock();
	cout << "Odpowiedz: " << answer << ". ";
	seq = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "CPU: Czas przetwarzania: " << seq << " ms" << endl;
	cout << "Przetwarzanie na GPU trwalo o " << par - seq << " ms dluzej." << endl << endl;


	query = { "0", "12", "22", "33", "41", "57", "66", "78", "80", "90" };
	c_start = std::clock();
	answer = classifier3.ask(query);
	c_end = std::clock();
	if (!answer.empty())
		cout << "Odpowiedz: " << answer << ". ";
	else
		cout << "Twoja odpowiedz nie zostala znaleziona." << endl;
	par = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "GPU: Czas przetwarzania: " << par << " ms" << endl;

	c_start = std::clock();
	answer = classifier3.sequentialAsk(query);
	c_end = std::clock();
	cout << "Odpowiedz: " << answer << ". ";
	seq = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "CPU: Czas przetwarzania: " << seq << " ms" << endl;
	cout << "Przetwarzanie na GPU trwalo o " << par - seq << " ms dluzej." << endl << endl;


	query = { "9", "16", "22", "30", "41", "56", "62", "73", "89", "93" };
	c_start = std::clock();
	answer = classifier3.ask(query);
	c_end = std::clock();
	if (!answer.empty())
		cout << "Odpowiedz: " << answer << ". ";
	else
		cout << "Twoja odpowiedz nie zostala znaleziona." << endl;
	par = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "GPU: Czas przetwarzania: " << par << " ms" << endl;

	c_start = std::clock();
	answer = classifier3.sequentialAsk(query);
	c_end = std::clock();
	cout << "Odpowiedz: " << answer << ". ";
	seq = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "CPU: Czas przetwarzania: " << seq << " ms" << endl;
	cout << "Przetwarzanie na GPU trwalo o " << par - seq << " ms dluzej." << endl << endl;

	////////////////////////////////////////////////////////////////////

	loadFromFile(classifier4, "plik4.txt");
	cout << "________________KLASYFIKATOR 4________________" << endl;
	classifier4.train();

	query = { "70", "188", "223", "304", "417", "531", "681", "779", "828", "948", "1035", "1151", "1226", "1344", "1427", "1520", "1646", "1741", "1805", "1916", "2000", "2117", "2201", "2373", "2423",
		"2572","2621","2786","2852","2916","3042","3135","3299","3371","3442","3591","3654","3742","3833","3987","4089","4113","4227","4396","4416","4588","4607","4778",
		"4828","4941","5099","5151","5290","5313","5489","5553","5686","5750","5880","5944","6010","6194","6227","6312","6405","6599","6652","6754","6801","6935","7066","7104", "7208",
		"7317","7426","7586","7690","7705","7857","7935","8064","8143","8258","8310","8417","8591","8644","8785","8882","8989","9005","9104","9261","9388","9477","9595","9686", "9715", "9891", "9916" };

	c_start = std::clock();
	answer = classifier4.ask(query);
	c_end = std::clock();
	if (!answer.empty())
		cout << "Odpowiedz: " << answer << ". ";
	else
		cout << "Twoja odpowiedz nie zostala znaleziona." << endl;
	par = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "GPU: Czas przetwarzania: " << par << " ms" << endl;

	c_start = std::clock();
	answer = classifier4.sequentialAsk(query);
	c_end = std::clock();
	cout << "Odpowiedz: " << answer << ". ";
	seq = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	cout << "CPU: Czas przetwarzania: " << seq << " ms" << endl;

	cout << "Przetwarzanie na GPU trwalo o " << par - seq << " ms dluzej." << endl << endl;


	cudaDeviceReset();
	system("pause");
	return 0;
}

/**
* Funkcja pozwalajaca na zaladowanie danych z pliku oraz zainicjowanie nimi klasyfikatora
*/
void loadFromFile(Classifier& classifier, string filename) {
	ifstream file(filename);
	//if (!file.is_open());
	//	return;
	string line, word, answer;
	vector<string> A;
	int attributes_number, examples_number, answers_number, counter;
	getline(file, line);
	
	attributes_number = atoi(line.c_str());
	//boost::lexical_cast<int>(line);
	for (int i = 0; i < attributes_number; ++i) {
		getline(file, line);
		stringstream ss(line);
		while (ss >> word)
			A.push_back(word);
		classifier.addCategory(A);
		A.clear();
	}
	getline(file, line);
	answers_number = atoi(line.c_str());
	//boost::lexical_cast<int>(line);
	for (int i = 0; i < answers_number; ++i) {
		getline(file, line);
		stringstream ss(line);
		while (ss >> answer)
			A.push_back(answer);
		classifier.addAnswers(A);
		A.clear();
	}
	getline(file, line);
	examples_number = atoi(line.c_str());
	//boost::lexical_cast<int>(line);
	for (int i = 0; i < examples_number; ++i) {
		counter = 0;
		getline(file, line);
		stringstream ss(line);
		while (ss >> word && counter != attributes_number) {
			A.push_back(word);
			counter++;
		}
		classifier.addTrainingExample(word, A);
		A.clear();
	}
	file.close();
	return;
}