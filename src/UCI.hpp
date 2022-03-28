#pragma once

#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <variant>


namespace UCI
{
    enum OptionType
    {
        CHECK,
        SPIN,
        COMBO,
        BUTTON,
        STRING
    };


    template<typename ...T>
    using OnChange = std::function<void(T...)>;
    using Stream = std::istringstream;


    class Option
    {
        const OptionType m_type;
        std::variant<bool*, int*, std::string*> m_data;
        std::variant<bool, int, std::string> m_default;
        int m_min;
        int m_max;
        std::vector<std::string> m_var;
        std::variant<OnChange<>, OnChange<bool>, OnChange<int>, OnChange<std::string>> m_change;

    public:
        Option(bool* data, bool def, OnChange<bool> change = nullptr)
            : m_type(CHECK), m_data(data), m_default(def), m_change(change)
        {
            *data = def;
        }
        Option(int* data, int def, int min, int max, OnChange<int> change = nullptr)
            : m_type(SPIN), m_data(data), m_default(def), m_min(min), m_max(max), m_change(change)
        {
            *data = def;
        }
        Option(std::string* data, std::string def, std::vector<std::string> var, OnChange<std::string> change = nullptr)
            : m_type(COMBO), m_data(data), m_default(def), m_var(var), m_change(change)
        {
            *data = def;
        }
        Option(std::string* data, std::string def, OnChange<std::string> change = nullptr)
            : m_type(STRING), m_data(data), m_default(def), m_change(change)
        {
            *data = def;
        }
        Option(OnChange<> change)
            : m_type(BUTTON), m_change(change)
        {}

        void set(std::string value);

        template<typename T>
        auto get() const { return std::get<T>(m_data); }

        friend std::ostream& operator<<(std::ostream& out, const Option& option);
    };
    std::ostream& operator<<(std::ostream& out, const Option& option);


    extern std::map<std::string, Option> OptionsMap;


    namespace Options
    {
        extern int Hash;
        extern int MultiPV;
        extern bool Ponder;
        extern int Threads;
    }


    void init_options();


    void main_loop();
    void setoption(Stream& stream);
    void uci(Stream& stream);
    void go(Stream& stream);
    void stop(Stream& stream);
    void quit(Stream& stream);
    void position(Stream& stream);
    void ponderhit(Stream& stream);
    void ucinewgame(Stream& stream);
    void isready(Stream& stream);


    Move move_from_uci(Position& position, std::string move_str);
}
