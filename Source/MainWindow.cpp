#include "MainWindow.h"
#include <QDebug>
#include <algorithm>
#include "PanStyle.h"
#include "ReconstructThread.h"
#include "ReconstructionPipeline.h"
#include "ui_MainWindow.h"
#include "AIProcessor.h"
#include "AIAssistant.h"
#include "CrosshairManager.h"
#include <QApplication>
#include <QProcess>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>
#include <QDateTime>
#include <QMessageBox>
#include <QProgressDialog>
#include <QSettings>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QMenu>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QScrollBar>

// VTK Includes
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkCellArray.h>
#include <vtkImageData.h>
#include <vtkImageReader2.h>
#include <vtkJPEGReader.h>
#include <vtkLight.h>
#include <vtkOBJImporter.h>
#include <vtkOBJReader.h>
#include <vtkPNGReader.h>
#include <vtkPlaneSource.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkTexture.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVertexGlyphFilter.h>
#include "Image2DLoader.h"
#include "Model3DLoader.h"
#include "DicomLoader.h"
#include <vtkCamera.h>
#include <vtkCornerAnnotation.h>
#include <vtkLineSource.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkActor2D.h>
#include <vtkProperty2D.h>
#include <vtkCoordinate.h>
#include <vtkTextProperty.h>
#include <vtkDICOMImageReader.h>

void MainWindow::clear3DModel() { for (auto &actor : modelActors) { renderer->RemoveActor(actor); } modelActors.clear(); }
void MainWindow::clear2DTexture() { if (texturePlaneActor) { renderer->RemoveActor(texturePlaneActor); texturePlaneActor = nullptr; } }
void MainWindow::clearPointCloud() { if (cloudActor) { renderer->RemoveActor(cloudActor); cloudActor = nullptr; pointCloudVisible = false; } }
void MainWindow::loadOBJwithMTL(const QString &objPath, const QString &mtlPath) {
    clear3DModel(); 
    auto actors = Model3DLoader::load(objPath, mtlPath);
    for (auto &a : actors) { modelActors.push_back(a); renderer->AddActor(a); }
    renderer->ResetCamera(); vtkWidget->renderWindow()->Render();
}
void MainWindow::resetToSingleRenderer() {
    if (m_crosshair) { m_crosshair->cleanup(); delete m_crosshair; m_crosshair = nullptr; }
    m_crosshairStyle = nullptr; auto *rw = vtkWidget->renderWindow();
    if (axialRenderer) { axialRenderer->RemoveAllViewProps(); rw->RemoveRenderer(axialRenderer); axialRenderer = nullptr; }
    if (sagittalRenderer) { sagittalRenderer->RemoveAllViewProps(); rw->RemoveRenderer(sagittalRenderer); sagittalRenderer = nullptr; }
    if (coronalRenderer) { coronalRenderer->RemoveAllViewProps(); rw->RemoveRenderer(coronalRenderer); coronalRenderer = nullptr; }
    renderer->SetViewport(0.0, 0.0, 1.0, 1.0); renderer->RemoveAllViewProps();
    vtkNew<PanStyle> style; if (rw->GetInteractor()) rw->GetInteractor()->SetInteractorStyle(style);
    renderer->SetBackground(0, 0, 0); rw->Render();
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow), pointCloudVisible(false), texturePlaneActor(nullptr), currentImageIndex(-1), currentAIMode(AIMode::None), isAutoNext(true) {
  ui->setupUi(this); autoTimer = new QTimer(this); connect(autoTimer, &QTimer::timeout, this, &MainWindow::onAutoTimerTimeout);
  
  aiAssistant = new AIAssistant(this);
  connect(aiAssistant, &AIAssistant::historyChanged, this, &MainWindow::updateChatUI);
  connect(aiAssistant, &AIAssistant::serverStatusChanged, this, &MainWindow::onAssistantStatusChanged);
  connect(aiAssistant, &AIAssistant::errorOccurred, this, &MainWindow::onAssistantError);

  QWidget *central = new QWidget(this); setCentralWidget(central); QVBoxLayout *mainLayout = new QVBoxLayout(central);
  vtkWidget = new QVTKOpenGLNativeWidget(this); mainLayout->addWidget(vtkWidget);
  setupNavigationUI();
  QString configPath = QApplication::applicationDirPath() + "/config.ini"; if (!QFile::exists(configPath)) configPath = QFileInfo(__FILE__).absolutePath() + "/config.ini";
  QSettings settings(configPath, QSettings::IniFormat); lastUsedPath = settings.value("Paths/lastUsedPath", "").toString();
  renderer = vtkSmartPointer<vtkRenderer>::New(); renderer->SetBackground(0, 0, 0); vtkWidget->renderWindow()->AddRenderer(renderer);
  vtkSmartPointer<vtkLight> headlight = vtkSmartPointer<vtkLight>::New(); headlight->SetLightTypeToHeadlight(); headlight->SetIntensity(1.5); renderer->AddLight(headlight);
  progressDialog = new QProgressDialog(this); progressBar = new QProgressBar(progressDialog); progressBar->setFixedHeight(50); progressBar->setMinimumWidth(400); progressDialog->setBar(progressBar); progressDialog->setCancelButton(nullptr); progressDialog->setRange(0, 100); progressDialog->reset(); progressDialog->setMinimumSize(500, 200);
  QToolBar *toolBar = addToolBar("Reconstruction 3d");
  QToolButton *btnP1 = new QToolButton(this); btnP1->setText("Phase 1: Viewer"); btnP1->setPopupMode(QToolButton::InstantPopup);
  QMenu *m1 = new QMenu(btnP1); m1->addAction("Load 2D images", this, &MainWindow::onLoad2DImages); m1->addAction("Load 3D images", this, &MainWindow::onLoad3DImages); m1->addAction("Load 2D DICOM", this, &MainWindow::onLoadDicom); btnP1->setMenu(m1); toolBar->addWidget(btnP1);
  toolBar->addSeparator();
  QToolButton *btnP2 = new QToolButton(this); btnP2->setText("Phase 2: Reconstruction"); btnP2->setPopupMode(QToolButton::InstantPopup);
  QMenu *m2 = new QMenu(btnP2); m2->addAction("Load multiple 2D images", this, &MainWindow::onLoadMultiple2DImages); m2->addAction("Run Reconstruction", this, &MainWindow::onRunReconstruction);
  QMenu *mc = m2->addMenu("Point Cloud"); actShowCloud = mc->addAction("Show Point Cloud", this, &MainWindow::onShowPointCloud); actHideCloud = mc->addAction("Hide Point Cloud", this, &MainWindow::onHidePointCloud); actHideCloud->setEnabled(false); btnP2->setMenu(m2); toolBar->addWidget(btnP2);
  toolBar->addSeparator();
  QToolButton *btnP4 = new QToolButton(this); btnP4->setText("Phase 4: AI Processing"); btnP4->setPopupMode(QToolButton::InstantPopup);
  QMenu *m4 = new QMenu(btnP4); m4->addAction("Train AI Model", this, &MainWindow::onTrainModel);
  QMenu *md = m4->addMenu("Object Detection"); actRunDet = md->addAction("Run Detection", this, &MainWindow::onObjectDetection); actHideDet = md->addAction("Hide Detection", this, &MainWindow::onHideAIResults); actHideDet->setEnabled(false);
  QMenu *ms = m4->addMenu("Segmentation"); actRunSeg = ms->addAction("Run Segmentation", this, &MainWindow::onSegmentation); actHideSeg = ms->addAction("Hide Segmentation", this, &MainWindow::onHideAIResults); actHideSeg->setEnabled(false); btnP4->setMenu(m4); toolBar->addWidget(btnP4);
  toolBar->addSeparator(); QAction *actChat = toolBar->addAction("AI Assistant"); connect(actChat, &QAction::triggered, this, &MainWindow::onToggleChatbot);
  setupChatbotUI(); reconstruction = new ReconstructionPipeline(); aiProcessor = new AIProcessor();
  QString modelsPath = QFileInfo(__FILE__).absolutePath() + "/../AITraining/Models"; aiProcessor->loadDetectionModel(modelsPath + "/yolo11n.onnx"); aiProcessor->loadSegmentationModel(modelsPath + "/yolo11n-seg.onnx");
  vtkRenderWindowInteractor *it = vtkWidget->renderWindow()->GetInteractor(); if (!it) { it = vtkRenderWindowInteractor::New(); vtkWidget->renderWindow()->SetInteractor(it); }
  vtkNew<PanStyle> style; it->SetInteractorStyle(style); it->Initialize(); renderer->ResetCamera();
  updateChatUI();
}

MainWindow::~MainWindow() { delete reconstruction; delete aiProcessor; delete ui; }

void MainWindow::onLoad2DImages() {
  QString fileName = QFileDialog::getOpenFileName(this, "Select 2D Image", lastUsedPath, "Images (*.png *.jpg *.jpeg *.bmp)"); if (fileName.isEmpty()) return;
  lastUsedPath = QFileInfo(fileName).absolutePath(); current2DImagePath = fileName;
  QSettings(QApplication::applicationDirPath() + "/config.ini", QSettings::IniFormat).setValue("Paths/lastUsedPath", lastUsedPath);
  resetToSingleRenderer(); clear3DModel(); clearPointCloud(); clear2DTexture(); currentAIMode = AIMode::None; updateMenuStates();
  texturePlaneActor = Image2DLoader::load(fileName);
  if (texturePlaneActor) {
      renderer->AddActor(texturePlaneActor); renderer->ResetCamera(); vtkWidget->renderWindow()->Render();
      QFileInfo fi(fileName); imageFileList = fi.dir().entryList(QStringList() << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp", QDir::Files, QDir::Name);
      currentImageIndex = imageFileList.indexOf(fi.fileName()); updateNavigationButtons();
  }
}
void MainWindow::onLoad3DImages() {
  QString obj = QFileDialog::getOpenFileName(this, "Select OBJ file", lastUsedPath, "OBJ Files (*.obj)"); if (obj.isEmpty()) return;
  QFileInfo fi(obj); lastUsedPath = fi.absolutePath(); clear3DModel(); clearPointCloud(); clear2DTexture(); resetToSingleRenderer();
  loadOBJwithMTL(obj, fi.path() + "/" + fi.completeBaseName() + ".mtl");
}
void MainWindow::onLoadMultiple2DImages() {
  QString startDir = lastUsedPath.isEmpty() ? "" : lastUsedPath;
  QStringList files = QFileDialog::getOpenFileNames(this, "Select Images", startDir, "Images (*.png *.jpg *.bmp)"); if (files.isEmpty()) return;
  lastUsedPath = QFileInfo(files.first()).absolutePath();
  QSettings(QApplication::applicationDirPath() + "/config.ini", QSettings::IniFormat).setValue("Paths/lastUsedPath", lastUsedPath);
  files.sort();
  std::vector<QString> paths; for (const auto &f : files) paths.push_back(f); reconstruction->setImages(paths);
  QFileInfo firstFile(files.first()); QString folder = firstFile.absolutePath(); QString baseName = firstFile.completeBaseName();
  QString prefix = baseName; int i = prefix.length() - 1; while (i >= 0 && prefix[i].isDigit()) --i;
  if (i >= 0) prefix = prefix.left(i + 1); else prefix.clear();
  QStringList possibleNames; if (!prefix.isEmpty()) { possibleNames << prefix + "_par.txt" << prefix.toLower() + "_par.txt"; }
  possibleNames << "temple_par.txt" << "templeR_par.txt" << "dino_par.txt" << "dinoR_par.txt" << "par.txt" << "camera_params.txt";
  QString loadedParamsPath = ""; for (const QString &name : possibleNames) {
    QString fullPath = folder + "/" + name; if (QFile::exists(fullPath)) { if (reconstruction->loadCameraParams(fullPath)) { loadedParamsPath = fullPath; break; } }
  }
  QString msg = QString("Đã tải %1 ảnh.").arg(files.size());
  if (!loadedParamsPath.isEmpty()) { msg += QString("\n\n✅ Camera params: %1").arg(QFileInfo(loadedParamsPath).fileName()); msg += "\n→ Sẽ dùng ground-truth projection matrices."; }
  else { msg += "\n\n⚠️ Không tìm thấy file params phù hợp trong:\n" + folder; msg += "\n→ Sẽ dùng estimated pose (kém chính xác hơn)."; }
  QMessageBox::information(this, "Load Images", msg);
}
void MainWindow::onRunReconstruction() {
  progressDialog->setLabelText("Đang tái tạo 3D..."); progressDialog->setRange(0, 0); progressDialog->setMinimumSize(500, 200); progressDialog->show(); QApplication::processEvents();
  ReconstructThread *thread = new ReconstructThread(reconstruction, this);
  connect(thread, &QThread::finished, this, [thread, this]() {
        progressDialog->hide(); bool success = thread->isSuccess(); int pointCount = (int)reconstruction->getPointCloud().size(); thread->deleteLater();
        if (!success || pointCount == 0) QMessageBox::warning(this, "Lỗi", "Reconstruction thất bại!\nKiểm tra lại dữ liệu nạp.");
        else { onShowPointCloud(); QMessageBox::information(this, "Thành công", QString("Reconstruction hoàn tất!\nTổng số điểm 3D: %1").arg(pointCount)); }
      }, Qt::QueuedConnection);
  thread->start();
}
void MainWindow::onShowPointCloud() {
  auto pts = reconstruction->getPointCloud(); auto colors = reconstruction->getPointColors(); if (pts.empty()) return;
  clear3DModel(); clearPointCloud(); resetToSingleRenderer();
  vtkNew<vtkPoints> vP; vtkNew<vtkCellArray> vV; vtkNew<vtkUnsignedCharArray> vC; vC->SetNumberOfComponents(3);
  for (size_t i = 0; i < pts.size(); ++i) { vP->InsertNextPoint(pts[i].x, pts[i].y, pts[i].z); vV->InsertNextCell(1); vV->InsertCellPoint(i); if (i < colors.size()) vC->InsertNextTuple3(colors[i][2], colors[i][1], colors[i][0]); else vC->InsertNextTuple3(255, 255, 255); }
  vtkNew<vtkPolyData> pd; pd->SetPoints(vP); pd->SetVerts(vV); pd->GetPointData()->SetScalars(vC);
  vtkNew<vtkVertexGlyphFilter> gf; gf->SetInputData(pd); vtkNew<vtkPolyDataMapper> m; m->SetInputConnection(gf->GetOutputPort());
  cloudActor = vtkSmartPointer<vtkActor>::New(); cloudActor->SetMapper(m); cloudActor->GetProperty()->SetPointSize(3);
  renderer->AddActor(cloudActor); renderer->ResetCamera(); vtkWidget->renderWindow()->Render(); pointCloudVisible = true; updateMenuStates();
}
void MainWindow::onHidePointCloud() { clearPointCloud(); updateMenuStates(); vtkWidget->renderWindow()->Render(); }
void MainWindow::onTrainModel() { QProcess::startDetached("cmd.exe", QStringList() << "/c" << "start" << "cmd.exe" << "/k" << "python" << QFileInfo(__FILE__).absolutePath() + "/../AITraining/TrainModel.py"); }
void MainWindow::onObjectDetection() {
  if (current2DImagePath.isEmpty() || !aiProcessor->isDetectionModelLoaded()) return;
  cv::Mat res = aiProcessor->runObjectDetection(cv::imread(current2DImagePath.toStdString()));
  QString tp = QApplication::applicationDirPath() + "/temp_ai_result.png"; cv::imwrite(tp.toStdString(), res);
  clear2DTexture(); texturePlaneActor = Image2DLoader::load(tp); if (texturePlaneActor) { renderer->AddActor(texturePlaneActor); vtkWidget->renderWindow()->Render(); }
  currentAIMode = AIMode::Detection; updateMenuStates();
}
void MainWindow::onSegmentation() {
  if (current2DImagePath.isEmpty() || !aiProcessor->isSegmentationModelLoaded()) return;
  cv::Mat res = aiProcessor->runSegmentation(cv::imread(current2DImagePath.toStdString()));
  QString tp = QApplication::applicationDirPath() + "/temp_ai_result.png"; cv::imwrite(tp.toStdString(), res);
  clear2DTexture(); texturePlaneActor = Image2DLoader::load(tp); if (texturePlaneActor) { renderer->AddActor(texturePlaneActor); vtkWidget->renderWindow()->Render(); }
  currentAIMode = AIMode::Segmentation; updateMenuStates();
}
void MainWindow::onHideAIResults() {
    currentAIMode = AIMode::None; if (!current2DImagePath.isEmpty()) { clear2DTexture(); texturePlaneActor = Image2DLoader::load(current2DImagePath); if (texturePlaneActor) { renderer->AddActor(texturePlaneActor); vtkWidget->renderWindow()->Render(); } }
    updateMenuStates();
}
void MainWindow::setupChatbotUI() {
    chatbotDock = new QDockWidget("AI Assistant", this); QWidget* dw = new QWidget(chatbotDock); QVBoxLayout* dl = new QVBoxLayout(dw);
    dl->setContentsMargins(10, 10, 10, 10); dl->setSpacing(10); QHBoxLayout* tl = new QHBoxLayout();
    modelSelector = new QComboBox(dw); modelSelector->addItem("Qwen2.5-3B-Instruct (Q4_K_M)"); modelSelector->addItem("Qwen2.5-3B-Instruct (Q8_0)"); modelSelector->addItem("Qwen2.5-7B-Instruct (Q4_K_M)");
    modelSelector->setCurrentIndex(QSettings(QFileInfo(__FILE__).absolutePath() + "/../Config/Config.ini", QSettings::IniFormat).value("AI/chatbot_model_index", 0).toInt());
    connect(modelSelector, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onModelSelected);
    tl->addWidget(modelSelector); btnNewChat = new QPushButton("New Chat", dw); connect(btnNewChat, &QPushButton::clicked, this, &MainWindow::onNewChat); tl->addWidget(btnNewChat);
    chatHistory = new QTextBrowser(dw); chatHistory->setHtml("<b>AI Assistant:</b> Chào bạn!"); chatHistory->setStyleSheet("background-color: #1e1e1e; color: #ffffff;");
    QHBoxLayout* il = new QHBoxLayout(); chatInput = new QLineEdit(dw); btnSendChat = new QPushButton("Gửi", dw); btnSendChat->setFixedSize(40, 40); il->addWidget(chatInput); il->addWidget(btnSendChat);
    dl->addLayout(tl); dl->addWidget(chatHistory); dl->addLayout(il); chatbotDock->setWidget(dw); addDockWidget(Qt::RightDockWidgetArea, chatbotDock); chatbotDock->hide();
    connect(btnSendChat, &QPushButton::clicked, this, &MainWindow::onSendChatMessage); connect(chatInput, &QLineEdit::returnPressed, this, &MainWindow::onSendChatMessage);
    chatHistory->setOpenLinks(false); connect(chatHistory, &QTextBrowser::anchorClicked, this, &MainWindow::onChatLinkClicked);
}
void MainWindow::onToggleChatbot() { if (chatbotDock->isHidden()) { chatbotDock->show(); if (!aiAssistant->isServerRunning()) aiAssistant->startServer(modelSelector->currentIndex()); } else chatbotDock->hide(); }
void MainWindow::onModelSelected(int index) { QSettings(QFileInfo(__FILE__).absolutePath() + "/../Config/Config.ini", QSettings::IniFormat).setValue("AI/chatbot_model_index", index); aiAssistant->startServer(index); }
void MainWindow::onAssistantStatusChanged(const QString &status) {
    QString h = chatHistory->toHtml();
    if (status == "Server AI đã sẵn sàng") h.replace("Đang khởi động Server...", "<font color='#00A36C'><b>" + status + "</b></font>");
    else chatHistory->append("<i>" + status + "</i>");
    if (status == "Server AI đã sẵn sàng") chatHistory->setHtml(h);
    chatHistory->moveCursor(QTextCursor::End);
}
void MainWindow::onAssistantError(const QString &error) { chatHistory->append("<font color='red'>" + error + "</font>"); }
void MainWindow::onSendChatMessage() { QString tx = chatInput->text().trimmed(); if (tx.isEmpty()) return; chatInput->clear(); aiAssistant->sendMessage(tx); btnSendChat->setEnabled(false); }
void MainWindow::updateChatUI() {
    btnSendChat->setEnabled(!aiAssistant->isThinking());
    chatHistory->clear(); 
    QString h = "<style>.msg { margin-bottom: 12px; padding: 8px; border-radius: 8px; }.user { background-color: #2b2d31; }.ai { background-color: #383a40; }.header { font-size: 11px; margin-bottom: 4px; display: flex; }.time { color: #888; font-size: 10px; margin-left: 10px; }.del { color: #ff4d4d; text-decoration: none; font-size: 10px; margin-left: auto; }.typing { color: #00A36C; font-style: italic; font-size: 12px; padding: 10px; }</style>";
    h += "<b>AI Assistant:</b> Chào bạn!<br><br>";
    auto history = aiAssistant->getHistory();
    for (int i = 0; i < history.size(); ++i) {
        QJsonObject m = history[i]; QString r = m["role"].toString();
        h += QString("<div class='msg %1'><div class='header'><b style='color: %2;'>%3</b><span class='time'>%4</span><a href='delete_%5' class='del'>[Xóa]</a></div><div class='content'>%6</div></div>")
             .arg((r == "user") ? "user" : "ai").arg((r == "user") ? "#0078D7" : "#00A36C").arg((r == "user") ? "Bạn" : "AI Assistant").arg(m["timestamp"].toString()).arg(i).arg(m["content"].toString());
    }
    if (aiAssistant->isThinking()) h += "<div class='typing'>AI đang suy nghĩ...</div>";
    chatHistory->setHtml(h);
    QTimer::singleShot(50, this, [this]() { chatHistory->verticalScrollBar()->setValue(chatHistory->verticalScrollBar()->maximum()); });
}
void MainWindow::onNewChat() { aiAssistant->newChat(); }
void MainWindow::onChatLinkClicked(const QUrl &url) {
    QString link = url.toString(); if (link.startsWith("delete_")) { int index = link.mid(7).toInt(); auto history = aiAssistant->getHistory(); if (index >= 0 && index < history.size()) { /* Logic xóa tin nhắn cụ thể cần thêm method vào AIAssistant nếu muốn, tạm thời gọi newChat hoặc ignore */ } }
}
void MainWindow::onLoadDicom() {
    QString fn = QFileDialog::getOpenFileName(this, "Select DICOM", lastUsedPath); if (fn.isEmpty()) return;
    lastUsedPath = QFileInfo(fn).absolutePath(); auto vd = DicomLoader::loadSeries(lastUsedPath); if (!vd || vd->GetNumberOfPoints() < 1) return;
    double r[2]; vd->GetScalarRange(r); clear3DModel(); clear2DTexture(); clearPointCloud(); renderer->RemoveAllViewProps();
    if (!axialRenderer) axialRenderer = vtkSmartPointer<vtkRenderer>::New(); if (!sagittalRenderer) sagittalRenderer = vtkSmartPointer<vtkRenderer>::New(); if (!coronalRenderer) coronalRenderer = vtkSmartPointer<vtkRenderer>::New();
    axialRenderer->SetViewport(0.0, 0.5, 0.5, 1.0); sagittalRenderer->SetViewport(0.5, 0.5, 1.0, 1.0); coronalRenderer->SetViewport(0.0, 0.0, 0.5, 0.5); renderer->SetViewport(0.5, 0.0, 1.0, 0.5);
    axialRenderer->SetBackground(0, 0, 0); sagittalRenderer->SetBackground(0, 0, 0); coronalRenderer->SetBackground(0, 0, 0); renderer->SetBackground(0, 0, 0);
    vtkWidget->renderWindow()->AddRenderer(axialRenderer); vtkWidget->renderWindow()->AddRenderer(sagittalRenderer); vtkWidget->renderWindow()->AddRenderer(coronalRenderer);
    auto addTitle = [](vtkRenderer* ren, const char* text) { vtkNew<vtkCornerAnnotation> ann; ann->SetText(2, text); ann->GetTextProperty()->SetColor(0.0, 1.0, 0.0); ann->GetTextProperty()->SetFontSize(16); ren->AddViewProp(ann); };
    addTitle(axialRenderer, "Axial"); addTitle(sagittalRenderer, "Sagittal"); addTitle(coronalRenderer, "Coronal"); addTitle(renderer, "3D View");
    renderer->AddViewProp(DicomLoader::createVolume(vd, r)); if (m_crosshair) { m_crosshair->cleanup(); delete m_crosshair; }
    m_crosshair = new CrosshairManager(this); m_crosshair->initialize(vd, sagittalRenderer, coronalRenderer, axialRenderer, vtkWidget->renderWindow(), 4746.0, 2373.0);
    auto style = vtkSmartPointer<CrosshairInteractorStyle>::New(); style->manager = m_crosshair; style->renderer3D = renderer; m_crosshairStyle = style;
    vtkWidget->renderWindow()->GetInteractor()->SetInteractorStyle(m_crosshairStyle);
    axialRenderer->ResetCamera(); sagittalRenderer->ResetCamera(); coronalRenderer->ResetCamera(); renderer->ResetCamera(); vtkWidget->renderWindow()->Render();
}
void MainWindow::setupNavigationUI() {
    QWidget *nw = new QWidget(this); QHBoxLayout *nl = new QHBoxLayout(nw); nw->setFixedHeight(40);
    btnPrev = new QPushButton("Prev", this); btnAutoPrev = new QPushButton("AutoPrev", this); btnAutoNext = new QPushButton("AutoNext", this); btnNext = new QPushButton("Next", this);
    btnAutoPrev->setCheckable(true); btnAutoNext->setCheckable(true);
    nl->addStretch(); nl->addWidget(btnPrev); nl->addWidget(btnAutoPrev); nl->addWidget(btnAutoNext); nl->addWidget(btnNext); nl->addStretch();
    qobject_cast<QVBoxLayout*>(centralWidget()->layout())->addWidget(nw);
    connect(btnPrev, &QPushButton::clicked, this, &MainWindow::onPrevImage); connect(btnNext, &QPushButton::clicked, this, &MainWindow::onNextImage);
    connect(btnAutoPrev, &QPushButton::clicked, this, &MainWindow::onAutoPrev); connect(btnAutoNext, &QPushButton::clicked, this, &MainWindow::onAutoNext);
}
void MainWindow::onNextImage() { if (currentImageIndex < imageFileList.size() - 1) { currentImageIndex++; loadCurrentIndexImage(); } }
void MainWindow::onPrevImage() { if (currentImageIndex > 0) { currentImageIndex--; loadCurrentIndexImage(); } }
void MainWindow::loadCurrentIndexImage() {
    current2DImagePath = QFileInfo(current2DImagePath).absolutePath() + "/" + imageFileList[currentImageIndex];
    clear2DTexture(); texturePlaneActor = Image2DLoader::load(current2DImagePath);
    if (texturePlaneActor) { renderer->AddActor(texturePlaneActor); if (currentAIMode == AIMode::Detection) onObjectDetection(); else if (currentAIMode == AIMode::Segmentation) onSegmentation(); vtkWidget->renderWindow()->Render(); }
    updateNavigationButtons();
}
void MainWindow::onAutoNext() { if (autoTimer->isActive() && isAutoNext) { autoTimer->stop(); btnAutoNext->setChecked(false); } else { isAutoNext = true; autoTimer->start(500); btnAutoNext->setChecked(true); btnAutoPrev->setChecked(false); } }
void MainWindow::onAutoPrev() { if (autoTimer->isActive() && !isAutoNext) { autoTimer->stop(); btnAutoPrev->setChecked(false); } else { isAutoNext = false; autoTimer->start(500); btnAutoPrev->setChecked(true); btnAutoNext->setChecked(false); } }
void MainWindow::onAutoTimerTimeout() {
    if (isAutoNext) { if (currentImageIndex < imageFileList.size() - 1) onNextImage(); else { autoTimer->stop(); btnAutoNext->setChecked(false); } }
    else { if (currentImageIndex > 0) onPrevImage(); else { autoTimer->stop(); btnAutoPrev->setChecked(false); } }
}
void MainWindow::updateNavigationButtons() { btnPrev->setEnabled(currentImageIndex > 0); btnNext->setEnabled(currentImageIndex >= 0 && currentImageIndex < imageFileList.size() - 1); }
void MainWindow::updateMenuStates() {
    actShowCloud->setEnabled(!pointCloudVisible); actHideCloud->setEnabled(pointCloudVisible);
    actRunDet->setEnabled(currentAIMode != AIMode::Detection); actHideDet->setEnabled(currentAIMode == AIMode::Detection);
    actRunSeg->setEnabled(currentAIMode != AIMode::Segmentation); actHideSeg->setEnabled(currentAIMode == AIMode::Segmentation);
}
