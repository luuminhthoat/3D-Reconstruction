#include "ReconstructThread.h"

ReconstructThread::ReconstructThread(ReconstructionPipeline *pipeline, QObject *parent)
    : QThread(parent), m_pipeline(pipeline), m_success(false)
{
}

void ReconstructThread::run()
{
    m_success = m_pipeline->reconstruct();
}
