#ifndef RECONSTRUCTTHREAD_H
#define RECONSTRUCTTHREAD_H

#include <QThread>
#include "ReconstructionPipeline.h"

class ReconstructThread : public QThread
{
    Q_OBJECT
public:
    explicit ReconstructThread(ReconstructionPipeline *pipeline, QObject *parent = nullptr);
    bool isSuccess() const { return m_success; }
protected:
    void run() override;
private:
    ReconstructionPipeline *m_pipeline;
    bool m_success;
};

#endif // RECONSTRUCTTHREAD_H
