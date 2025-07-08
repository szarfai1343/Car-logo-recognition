#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace cv;
using namespace dnn;

// �ADOWANIE KLAS
vector<string> load_class_list(const string& class_file_path) {
    vector<string> class_list;
    ifstream ifs(class_file_path);
    string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

// �ADOWANIE MODELU
void load_net(Net& net, const string& model_path, bool is_cuda) {
    net = readNet(model_path); //net to obiekt modelu sieci neuronowej �adowany za pomoc� funkcji
    if (is_cuda) { //je�li jest u�ywa GPU (szybsze dzia�anie)
        cout << "Using CUDA\n";
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
    }
    else { //je�li nie u�ywa CPU
        cout << "Using CPU\n";
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
}

// PROGI PEWNO�CI
//rozmiary obrazu wej�ciowego, do kt�rego modeloczekuje dopasowania (nale�y dobra� zgodnie z modelem)
const float INPUT_WIDTH = 640.0f;
const float INPUT_HEIGHT = 640.0f;

const float SCORE_THRESHOLD_CAR = 0.2f; //minimalna wpr�g praawdopodobie�stwa dla klasy aby detekcja zosta�a uznana za wa�n�
const float NMS_THRESHOLD_CAR = 0.4f;   //pr�g u�ywany podczas eliminacji nak�adaja�ych si� ramek
const float CONFIDENCE_THRESHOLD_CAR = 0.4f;    //minimalny pr�g pewno�ci detkecji aby ramka by�a uznana za poprawnie wykryt�

// pr�g pewno�ci dla tablic rejestracyjnych zmniejszony poniewa� model jest s�abej wytrenowany i nie radzi sobie tak dobrze co sprawia �e pojawiaj� si� fa�szywe detekcje
const float SCORE_THRESHOLD_PLATES = 0.5f; 
const float NMS_THRESHOLD_PLATES = 0.3f;  
const float CONFIDENCE_THRESHOLD_PLATES = 0.5f;

// WYKRYCIE = BOX + NAZWA KLASY + PEWNO��
struct Detection {
    int class_id;
    float confidence;
    Rect box;
};

// PRZEFORMATWOWANIE OBRAZU NA KWADRAT (POTRZEBNE DLA YOLO):
// pobiera wymiary obrazu, tworzy kwadratowy obraz i kopiuje orginalny obraz do rogu nowego
Mat format_yolov5(const Mat& source) { //obraz wej�ciowy w postaci macierzy, klatka z kamery
    int col = source.cols;  //szeroko�� obrazu
    int row = source.rows;  //wysoko�� obrazu
    int _max = max(col, row);   //znajdujemy wi�kszy z wymiar�w aby przeskalowa� obraz do kwadratu (wymaganie yolov5)
    Mat result = Mat::zeros(_max, _max, CV_8UC3);   //pusta macierz (czarne t�o) o wymiarach MAXxMAX, CV_8U 8bitowa liczb ca�kowita, C3=RGB
    source.copyTo(result(Rect(0, 0, col, row)));    //kopiowanie obrazu do kwadraowej macierzy
    return result;
}

// DETEKCJA OBIEKT�W
void detect(Mat& image, Net& net, vector<Detection>& output, const vector<string>& class_list, float score_threshold, float nms_threshold, float confidence_threshold) {
    Mat blob;
    auto input_image = format_yolov5(image);
    //funkcja openCV blobFromImage: obraz wej�ciowy jest skalowany. Warto�ci pikseli s� dzielone przez 255 co sprowadza je do zakresu 0-1 (potrzebna normalizacja dla YOLOv5)
    //Size to 640 na 640 (zale�ne od modelu i w jaki spos�b rozpocz�to deep learning)
    //warto�� true sprwia �e zostaje u�yty BGR jako kolejno�� kana��w (dom�lne dla openCV)
    //Funkcja przekszta�ca obraz w blob kt�ry model jest w stanie przetworzy�
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outputs;
    //funkcja forward sprawia �e model przetwarza dane wej�ciwe i zapisuje je do outputs
    //funckja getUnconnectedOutLayersNames pobiera nazwy warst wyj�ciwoych modelu kt�re YOLOv5 ma zdefiniowane, produkuj� one informacje o wsp�rz�dnych ramek i prawdopodobie�stwie klasy
    net.forward(outputs, net.getUnconnectedOutLayersNames()); 
    
    //pobieramy liczb� wymiar�w dla wykrytego obiektu
    //pierwsze 4 wymiary opisuj� wsp�rz�dne ramki x y w h
    //pi�ty wymiar to pewno�� wykrycia
    //reszta wymiar�w zale�y od treningu poniewa� s� to prawdopodobie�stwa dla klas (Liczb� wymiar�w mo�na sprawdzi� w Netron.app oraz ustawi� na "sztywno"
    const int dimensions = outputs[0].size[2];
    
    //Pobiera liczb� wierszy danych wyj�ciwoych co odpowiada liczbie potencjalnych wykry� (dla YOLOv5 jest to przewa�nie 25200 co mo�na sprawdzi� w Netron.app
    const int rows = outputs[0].size[1];
    
    //Oblicza wsp�czynnik skalowania dla osi x i y
    //YOLOv% wymaga aby obrazy wej�iowe mia�y sta�y rozmiar, taki jaki zosta� ustalony przy uczeniu (w moim przypadku 640x640)
    //x oraz y factor to stosunki szeroko�ci i wysoko�ci orginalnego obrazu do rozmiaru wej�ciowego modelu
    //Po obliczeniu predkkcji wyniki musz� zosta� przeskalowane spowrotem do orgyginalych wymiar�w
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    //Pobiera wska�nik do danych wyj�ciwoych, poniewa� YOLOv5 generuje dane w formie macierzy, wi�c aby je odczyta� potrzebujemy go
    float* data = (float*)outputs[0].data;
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4]; // pobieramy warto�� pewno�ci wykrycia (przedzia� 0,1) (5 wymiar to pewno�� wykrycia)
        if (confidence >= confidence_threshold) { //sprawdzamy czy pewno�� wykrycia przekracza zdefiniowany pr�g (wyklucza to zbyt niepewne detekcje)
            float* classes_scores = data + 5; //przesuwamy wska�nik do miejsca gdzie zaczynaj� si� wyniki klas
            Mat scores(1, class_list.size(), CV_32FC1, classes_scores); //tworzymy macierz kt�ra przechowuje prawdopodobie�stwa dla klas co u�atwia operacje
            Point class_id_point;
            double max_class_score; 
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point); //funkcja znajduje klase o najwy�szym prawdopodobie�stwie kt�ra zostanie wy�wietlona

            if (max_class_score > score_threshold) {   //Sprawdzamy czy prawdopodobie�stwo dla klasy przekracza ustalony pr�g (eliminacja fa�szywych detekcji)
                int class_id = class_id_point.x;
                
                float x = data[0];  //�rodek ramki w osi X
                float y = data[1];  //�rodek ramki w osi Y
                float w = data[2];  //szerko�� ramki
                float h = data[3];  //wysoko�� ramki

                //Dzi�ki pobranym wy�ej parametrom, u�ywaj�c wcze�niej zadeklaorwanych x oraz y factor, mo�emy przeskalowa� obraz do oryginalnych wymiar�w
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                //Wyniki wykrycia przechowaywane s� co jest przydatne w dalszym przetwarzaniu
                class_ids.push_back(class_id);
                confidences.push_back(confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        data += dimensions; //przesuwamy wska�nik do kolejnego wiersza wyj�cia
    }

    // Non-Maximum Suppression: SPRAWIA �E BOXY NIE NAK�ADAJ� SI� NA TEN SAM ELEMENT CZYNI�C DETEKCJE NIE CZYTELN�
    vector<int> indices;

    //boxes reprezentuje ramki wykryte przez model (pozycja i rozmiar)
    //confidences wektor pewno�ci dla ramki
    //score_threshold to minimalny pr�g pewno�ci, pewno�� mniejsza ni� ta zostaje ignorowane przed uruchomieniem NMS
    //nms_threshold to warto�c okre�laj�ca maksymalny stopie� nak�adania si� ramke
    //w indices przechowujemy indeksy ramek kt�re zosta�y zachowane po NMS
    NMSBoxes(boxes, confidences, score_threshold, nms_threshold, indices);

    //przechowujemy ko�cowe wyniki detekcji ramek kt�re przesz�y przez NMS
    for (int idx : indices) {
        output.push_back({ class_ids[idx], confidences[idx], boxes[idx] });
    }
}

// MAIN
int main(int argc, char** argv) {

    //Pod zmienne przypisujemy �cie�ki do plik�w zawieraj�cych wagi model�w
    string class_file1 = "config_files/classes1.txt";
    string class_file2 = "config_files/rej.txt";

    //Pod zmienne przypisujemy �cie�ki do plik�w zawieraj�cych wagi model�w
    //Modele musza by� w formacie onnx poniewa� tylko taki obs�uguje C++ 
    string model_file1 = "config_files/best.onnx";
    string model_file2 = "config_files/rej.onnx";

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0; //sprawdzamy czy program ma zosta� uruchominoy z wykorzystaniem GPU czy CPU

    vector<string> class_list1 = load_class_list(class_file1);
    vector<string> class_list2 = load_class_list(class_file2);


    //Deklarujemy 2 typy obiekt�w sieci neuronowej przy u�yciu �cie�ek do pliku wag, oraz konfigurujemy je do pracy przy pomocy GPU lub CPU za pomoc� is_cuda
    Net net1, net2;
    load_net(net1, model_file1, is_cuda);
    load_net(net2, model_file2, is_cuda);

    //Obs�uga kamery w open CV
    Mat frame;
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cerr << "Error opening video file\n";
        return -1;
    }

    while (true) {
        capture.read(frame);
        if (frame.empty()) {
            cout << "End of stream\n";
            break;
        }

        
        vector<Detection> detections1, detections2;
        //Przy pomocy funkcji detect wykonujemy detekcje obiekt�w w obrazie przy u�yciu modeli sieci neuronowych
        //frame odpowiada za aktualn� klatk� obrazu z kamery
        detect(frame, net1, detections1, class_list1, SCORE_THRESHOLD_CAR, NMS_THRESHOLD_CAR, CONFIDENCE_THRESHOLD_CAR);
        detect(frame, net2, detections2, class_list2, SCORE_THRESHOLD_PLATES, NMS_THRESHOLD_PLATES, CONFIDENCE_THRESHOLD_PLATES);

        //iteracja przez wyniki detekcji naszych modeli i wyrysowanie na obrazie odpowiednich ramek oraz podpis�w dla wykrytych obiekt�w (wizualizacja wynik�w)
        for (const auto& detection : detections1) {
            rectangle(frame, detection.box, Scalar(0, 255, 0), 2);  //rysowanie ramek (frame obraz na kt�rym rysujemy, detection.box pozycja i rozmiar ramki, kolor zielony, grubu��)
            string label = class_list1[detection.class_id] + ": " + format("%.2f", detection.confidence);   //tworzenie etykiety (nazwa klasy obiektu, zformatowana pewno�� detekcji)
            putText(frame, label, Point(detection.box.x, detection.box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2); //wyrysowanie etykiety na obrazie (pozycja etykiety,
                                                                                                                                  //czcionka, rozmiar czcionki, kolor, grubo�c lini
        }

        for (const auto& detection : detections2) {
            rectangle(frame, detection.box, Scalar(255, 0, 0), 2);
            string label = class_list2[detection.class_id] + ": " + format("%.2f", detection.confidence);
            putText(frame, label, Point(detection.box.x, detection.box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
        }


        //wy�wietlenie obrazu w oknie
        imshow("Marcin Pi�ta projekt", frame);
        if (waitKey(1) != -1) {
            break;
        }
    }

    return 0;
}

