#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace cv;
using namespace dnn;

// £ADOWANIE KLAS
vector<string> load_class_list(const string& class_file_path) {
    vector<string> class_list;
    ifstream ifs(class_file_path);
    string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

// £ADOWANIE MODELU
void load_net(Net& net, const string& model_path, bool is_cuda) {
    net = readNet(model_path); //net to obiekt modelu sieci neuronowej ³adowany za pomoc¹ funkcji
    if (is_cuda) { //jeœli jest u¿ywa GPU (szybsze dzia³anie)
        cout << "Using CUDA\n";
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
    }
    else { //jeœli nie u¿ywa CPU
        cout << "Using CPU\n";
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
}

// PROGI PEWNOŒCI
//rozmiary obrazu wejœciowego, do którego modeloczekuje dopasowania (nale¿y dobraæ zgodnie z modelem)
const float INPUT_WIDTH = 640.0f;
const float INPUT_HEIGHT = 640.0f;

const float SCORE_THRESHOLD_CAR = 0.2f; //minimalna wpróg praawdopodobieñstwa dla klasy aby detekcja zosta³a uznana za wa¿n¹
const float NMS_THRESHOLD_CAR = 0.4f;   //próg u¿ywany podczas eliminacji nak³adajaæych siê ramek
const float CONFIDENCE_THRESHOLD_CAR = 0.4f;    //minimalny próg pewnoœci detkecji aby ramka by³a uznana za poprawnie wykryt¹

// próg pewnoœci dla tablic rejestracyjnych zmniejszony poniewa¿ model jest s³abej wytrenowany i nie radzi sobie tak dobrze co sprawia ¿e pojawiaj¹ siê fa³szywe detekcje
const float SCORE_THRESHOLD_PLATES = 0.5f; 
const float NMS_THRESHOLD_PLATES = 0.3f;  
const float CONFIDENCE_THRESHOLD_PLATES = 0.5f;

// WYKRYCIE = BOX + NAZWA KLASY + PEWNOŒÆ
struct Detection {
    int class_id;
    float confidence;
    Rect box;
};

// PRZEFORMATWOWANIE OBRAZU NA KWADRAT (POTRZEBNE DLA YOLO):
// pobiera wymiary obrazu, tworzy kwadratowy obraz i kopiuje orginalny obraz do rogu nowego
Mat format_yolov5(const Mat& source) { //obraz wejœciowy w postaci macierzy, klatka z kamery
    int col = source.cols;  //szerokoœæ obrazu
    int row = source.rows;  //wysokoœæ obrazu
    int _max = max(col, row);   //znajdujemy wiêkszy z wymiarów aby przeskalowaæ obraz do kwadratu (wymaganie yolov5)
    Mat result = Mat::zeros(_max, _max, CV_8UC3);   //pusta macierz (czarne t³o) o wymiarach MAXxMAX, CV_8U 8bitowa liczb ca³kowita, C3=RGB
    source.copyTo(result(Rect(0, 0, col, row)));    //kopiowanie obrazu do kwadraowej macierzy
    return result;
}

// DETEKCJA OBIEKTÓW
void detect(Mat& image, Net& net, vector<Detection>& output, const vector<string>& class_list, float score_threshold, float nms_threshold, float confidence_threshold) {
    Mat blob;
    auto input_image = format_yolov5(image);
    //funkcja openCV blobFromImage: obraz wejœciowy jest skalowany. Wartoœci pikseli s¹ dzielone przez 255 co sprowadza je do zakresu 0-1 (potrzebna normalizacja dla YOLOv5)
    //Size to 640 na 640 (zale¿ne od modelu i w jaki sposób rozpoczêto deep learning)
    //wartoœæ true sprwia ¿e zostaje u¿yty BGR jako kolejnoœæ kana³ów (domœlne dla openCV)
    //Funkcja przekszta³ca obraz w blob który model jest w stanie przetworzyæ
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    net.setInput(blob);

    vector<Mat> outputs;
    //funkcja forward sprawia ¿e model przetwarza dane wejœciwe i zapisuje je do outputs
    //funckja getUnconnectedOutLayersNames pobiera nazwy warst wyjœciwoych modelu które YOLOv5 ma zdefiniowane, produkuj¹ one informacje o wspó³rzêdnych ramek i prawdopodobieñstwie klasy
    net.forward(outputs, net.getUnconnectedOutLayersNames()); 
    
    //pobieramy liczbê wymiarów dla wykrytego obiektu
    //pierwsze 4 wymiary opisuj¹ wspó³rzêdne ramki x y w h
    //pi¹ty wymiar to pewnoœæ wykrycia
    //reszta wymiarów zale¿y od treningu poniewa¿ s¹ to prawdopodobieñstwa dla klas (Liczbê wymiarów mo¿na sprawdziæ w Netron.app oraz ustawiæ na "sztywno"
    const int dimensions = outputs[0].size[2];
    
    //Pobiera liczbê wierszy danych wyjœciwoych co odpowiada liczbie potencjalnych wykryæ (dla YOLOv5 jest to przewa¿nie 25200 co mo¿na sprawdziæ w Netron.app
    const int rows = outputs[0].size[1];
    
    //Oblicza wspó³czynnik skalowania dla osi x i y
    //YOLOv% wymaga aby obrazy wejœiowe mia³y sta³y rozmiar, taki jaki zosta³ ustalony przy uczeniu (w moim przypadku 640x640)
    //x oraz y factor to stosunki szerokoœci i wysokoœci orginalnego obrazu do rozmiaru wejœciowego modelu
    //Po obliczeniu predkkcji wyniki musz¹ zostaæ przeskalowane spowrotem do orgyginalych wymiarów
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    //Pobiera wskaŸnik do danych wyjœciwoych, poniewa¿ YOLOv5 generuje dane w formie macierzy, wiêc aby je odczytaæ potrzebujemy go
    float* data = (float*)outputs[0].data;
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4]; // pobieramy wartoœæ pewnoœci wykrycia (przedzia³ 0,1) (5 wymiar to pewnoœæ wykrycia)
        if (confidence >= confidence_threshold) { //sprawdzamy czy pewnoœæ wykrycia przekracza zdefiniowany próg (wyklucza to zbyt niepewne detekcje)
            float* classes_scores = data + 5; //przesuwamy wskaŸnik do miejsca gdzie zaczynaj¹ siê wyniki klas
            Mat scores(1, class_list.size(), CV_32FC1, classes_scores); //tworzymy macierz która przechowuje prawdopodobieñstwa dla klas co u³atwia operacje
            Point class_id_point;
            double max_class_score; 
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point); //funkcja znajduje klase o najwy¿szym prawdopodobieñstwie która zostanie wyœwietlona

            if (max_class_score > score_threshold) {   //Sprawdzamy czy prawdopodobieñstwo dla klasy przekracza ustalony próg (eliminacja fa³szywych detekcji)
                int class_id = class_id_point.x;
                
                float x = data[0];  //œrodek ramki w osi X
                float y = data[1];  //œrodek ramki w osi Y
                float w = data[2];  //szerkoœæ ramki
                float h = data[3];  //wysokoœæ ramki

                //Dziêki pobranym wy¿ej parametrom, u¿ywaj¹c wczeœniej zadeklaorwanych x oraz y factor, mo¿emy przeskalowaæ obraz do oryginalnych wymiarów
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                //Wyniki wykrycia przechowaywane s¹ co jest przydatne w dalszym przetwarzaniu
                class_ids.push_back(class_id);
                confidences.push_back(confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        data += dimensions; //przesuwamy wskaŸnik do kolejnego wiersza wyjœcia
    }

    // Non-Maximum Suppression: SPRAWIA ¯E BOXY NIE NAK£ADAJ¥ SIÊ NA TEN SAM ELEMENT CZYNI¥C DETEKCJE NIE CZYTELN¥
    vector<int> indices;

    //boxes reprezentuje ramki wykryte przez model (pozycja i rozmiar)
    //confidences wektor pewnoœci dla ramki
    //score_threshold to minimalny próg pewnoœci, pewnoœæ mniejsza ni¿ ta zostaje ignorowane przed uruchomieniem NMS
    //nms_threshold to wartoœc okreœlaj¹ca maksymalny stopieñ nak³adania siê ramke
    //w indices przechowujemy indeksy ramek które zosta³y zachowane po NMS
    NMSBoxes(boxes, confidences, score_threshold, nms_threshold, indices);

    //przechowujemy koñcowe wyniki detekcji ramek które przesz³y przez NMS
    for (int idx : indices) {
        output.push_back({ class_ids[idx], confidences[idx], boxes[idx] });
    }
}

// MAIN
int main(int argc, char** argv) {

    //Pod zmienne przypisujemy œcie¿ki do plików zawieraj¹cych wagi modelów
    string class_file1 = "config_files/classes1.txt";
    string class_file2 = "config_files/rej.txt";

    //Pod zmienne przypisujemy œcie¿ki do plików zawieraj¹cych wagi modelów
    //Modele musza byæ w formacie onnx poniewa¿ tylko taki obs³uguje C++ 
    string model_file1 = "config_files/best.onnx";
    string model_file2 = "config_files/rej.onnx";

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0; //sprawdzamy czy program ma zostaæ uruchominoy z wykorzystaniem GPU czy CPU

    vector<string> class_list1 = load_class_list(class_file1);
    vector<string> class_list2 = load_class_list(class_file2);


    //Deklarujemy 2 typy obiektów sieci neuronowej przy u¿yciu œcie¿ek do pliku wag, oraz konfigurujemy je do pracy przy pomocy GPU lub CPU za pomoc¹ is_cuda
    Net net1, net2;
    load_net(net1, model_file1, is_cuda);
    load_net(net2, model_file2, is_cuda);

    //Obs³uga kamery w open CV
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
        //Przy pomocy funkcji detect wykonujemy detekcje obiektów w obrazie przy u¿yciu modeli sieci neuronowych
        //frame odpowiada za aktualn¹ klatkê obrazu z kamery
        detect(frame, net1, detections1, class_list1, SCORE_THRESHOLD_CAR, NMS_THRESHOLD_CAR, CONFIDENCE_THRESHOLD_CAR);
        detect(frame, net2, detections2, class_list2, SCORE_THRESHOLD_PLATES, NMS_THRESHOLD_PLATES, CONFIDENCE_THRESHOLD_PLATES);

        //iteracja przez wyniki detekcji naszych modeli i wyrysowanie na obrazie odpowiednich ramek oraz podpisów dla wykrytych obiektów (wizualizacja wyników)
        for (const auto& detection : detections1) {
            rectangle(frame, detection.box, Scalar(0, 255, 0), 2);  //rysowanie ramek (frame obraz na którym rysujemy, detection.box pozycja i rozmiar ramki, kolor zielony, grubuœæ)
            string label = class_list1[detection.class_id] + ": " + format("%.2f", detection.confidence);   //tworzenie etykiety (nazwa klasy obiektu, zformatowana pewnoœæ detekcji)
            putText(frame, label, Point(detection.box.x, detection.box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2); //wyrysowanie etykiety na obrazie (pozycja etykiety,
                                                                                                                                  //czcionka, rozmiar czcionki, kolor, gruboœc lini
        }

        for (const auto& detection : detections2) {
            rectangle(frame, detection.box, Scalar(255, 0, 0), 2);
            string label = class_list2[detection.class_id] + ": " + format("%.2f", detection.confidence);
            putText(frame, label, Point(detection.box.x, detection.box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
        }


        //wyœwietlenie obrazu w oknie
        imshow("Marcin Piêta projekt", frame);
        if (waitKey(1) != -1) {
            break;
        }
    }

    return 0;
}

