#ifndef AIASSISTANT_H
#define AIASSISTANT_H

#include <QObject>
#include <QProcess>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QList>
#include <QJsonObject>
#include <QJsonArray>
#include <QStringList>

class AIAssistant : public QObject {
    Q_OBJECT
public:
    explicit AIAssistant(QObject *parent = nullptr);
    ~AIAssistant();

    void startServer(int modelIndex);
    void stopServer();
    void sendMessage(const QString &text);
    void newChat();
    
    QList<QJsonObject> getHistory() const { return conversationHistory; }
    bool isThinking() const { return m_isThinking; }
    bool isServerRunning() const { return aiServerProcess->state() != QProcess::NotRunning; }

signals:
    void historyChanged();
    void serverStatusChanged(const QString &status);
    void errorOccurred(const QString &error);
    void responseReceived();

private slots:
    void onProcessReadyRead();
    void onProcessError();
    void onReplyFinished(QNetworkReply* reply);

private:
    QProcess *aiServerProcess;
    QNetworkAccessManager *networkManager;
    QList<QJsonObject> conversationHistory;
    bool m_isThinking = false;

    void saveHistory();
    void loadHistory();
    QString getHistoryPath();
};

#endif // AIASSISTANT_H
