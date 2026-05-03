#ifndef RECONSTRUCTIONWORKER_H
#define RECONSTRUCTIONWORKER_H

#endif // RECONSTRUCTIONWORKER_H

#include <QThread>
#include "ReconstructionPipeline.h"cdefs.h"

class ReconstructionWorker : public QObject
{
    Q_OBJECT
public:
    ReconstructionPipeline *pipeline;
    bool success;
public slots:
    void run() {
        success = pipeline->reconstruct();
        emit finished(success);
    }
signals:
    void finished(bool success);
};
