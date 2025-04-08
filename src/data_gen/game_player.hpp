#pragma once

#include "Types.hpp"
#include "Position.hpp"
#include "data_gen/data_gen.hpp"
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

template <typename T>
class ThreadSafeQueue
{
    std::queue<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cvar;
    std::atomic_size_t m_size;

public:
    ThreadSafeQueue() : m_size(0) {}

    void push(const T& item)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_queue.push(item);
        m_cvar.notify_one();
    }

    void push(T&& item)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_queue.push(item);
        m_size++;
        m_cvar.notify_one();
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cvar.wait(lock, [this]() { return !m_queue.empty(); });

        T item = m_queue.front();
        m_queue.pop();
        m_size--;
        return item;
    }

    bool empty()
    {
        return m_size.load() == 0;
    }
};

namespace GamePlayer
{
    void play_games(std::istringstream& stream);

    void games_to_epd(std::istringstream& stream);

    void check_games(std::istringstream& stream);

    void repair_games(std::istringstream& stream);

    bool file_valid(std::string filename);
}