#include "AIAssistant.h"
#include <QDebug>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QDateTime>
#include <QDir>
#include <QApplication>

AIAssistant::AIAssistant(QObject *parent) : QObject(parent) {
    aiServerProcess = new QProcess(this);
    networkManager = new QNetworkAccessManager(this);

    connect(aiServerProcess, &QProcess::readyReadStandardOutput, this, &AIAssistant::onProcessReadyRead);
    connect(aiServerProcess, &QProcess::readyReadStandardError, this, &AIAssistant::onProcessError);
    connect(networkManager, &QNetworkAccessManager::finished, this, &AIAssistant::onReplyFinished);

    loadHistory();
}

AIAssistant::~AIAssistant() {
    stopServer();
}

void AIAssistant::startServer(int modelIndex) {
    stopServer();
    
    QString sp = QFileInfo(__FILE__).absolutePath() + "/../AITraining/StartChatbotServer.py";
    if (QFileInfo::exists(sp)) {
        emit serverStatusChanged("Đang khởi động Server...");
        aiServerProcess->start("python", QStringList() << sp << QString::number(modelIndex));
    } else {
        emit errorOccurred("Không tìm thấy file StartChatbotServer.py");
    }
}

void AIAssistant::stopServer() {
    if (aiServerProcess->state() != QProcess::NotRunning) {
#ifdef Q_OS_WIN
        QProcess::execute("taskkill", QStringList() << "/F" << "/T" << "/PID" << QString::number(aiServerProcess->processId()));
#else
        aiServerProcess->kill();
#endif
        aiServerProcess->waitForFinished(3000);
    }
}

void AIAssistant::sendMessage(const QString &text) {
    if (text.isEmpty()) return;

    QJsonObject um;
    um["role"] = "user";
    um["content"] = text;
    um["timestamp"] = QDateTime::currentDateTime().toString("HH:mm");
    
    conversationHistory.append(um);
    saveHistory();
    emit historyChanged();

    QNetworkRequest req(QUrl("http://127.0.0.1:8080/v1/chat/completions"));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QJsonArray msgs;
    for (const auto& m : conversationHistory) msgs.append(m);

    QJsonObject js;
    js["messages"] = msgs;
    js["temperature"] = 0.7;
    js["max_tokens"] = 512;

    networkManager->post(req, QJsonDocument(js).toJson());
    m_isThinking = true;
    emit historyChanged();
}

void AIAssistant::newChat() {
    conversationHistory.clear();
    saveHistory();
    emit historyChanged();
}

void AIAssistant::onProcessReadyRead() {
    QString out = aiServerProcess->readAllStandardOutput();
    qDebug() << "[AI Assistant Server]" << out;
    if (out.contains("Uvicorn running on") || out.contains("Application startup complete")) {
        emit serverStatusChanged("Server AI đã sẵn sàng");
    }
}

void AIAssistant::onProcessError() {
    QString err = aiServerProcess->readAllStandardError();
    qDebug() << "[AI Assistant Server Error]" << err;
    if (err.contains("Uvicorn running on") || err.contains("Application startup complete")) {
        emit serverStatusChanged("Server AI đã sẵn sàng");
    }
}

void AIAssistant::onReplyFinished(QNetworkReply* reply) {
    m_isThinking = false;
    if (reply->error() == QNetworkReply::NoError) {
        QJsonObject m = QJsonDocument::fromJson(reply->readAll()).object()["choices"].toArray()[0].toObject()["message"].toObject();
        QJsonObject am;
        am["role"] = "assistant";
        am["content"] = m["content"].toString();
        am["timestamp"] = QDateTime::currentDateTime().toString("HH:mm");
        
        conversationHistory.append(am);
        saveHistory();
        emit responseReceived();
        emit historyChanged();
    } else {
        emit errorOccurred("Lỗi kết nối. Đang thử lại...");
    }
    reply->deleteLater();
}

void AIAssistant::saveHistory() {
    QFile f(getHistoryPath());
    if (f.open(QIODevice::WriteOnly)) {
        QJsonArray a;
        for (const auto& m : conversationHistory) a.append(m);
        f.write(QJsonDocument(a).toJson());
        f.close();
    }
}

void AIAssistant::loadHistory() {
    QFile f(getHistoryPath());
    if (f.open(QIODevice::ReadOnly)) {
        QJsonArray a = QJsonDocument::fromJson(f.readAll()).array();
        conversationHistory.clear();
        for (auto v : a) conversationHistory.append(v.toObject());
        f.close();
        emit historyChanged();
    }
}

QString AIAssistant::getHistoryPath() {
    return QFileInfo(__FILE__).absolutePath() + "/../Config/chat_history.json";
}
