#pragma once

#include <iterator>


namespace PDE
{
    template<typename T, typename Iterator> 
    struct Span
    {
        Iterator first;
        Iterator last;
 
        T &operator[](int i) const
        { return *(first + i); }

        Iterator begin() const
        { return first; }

        Iterator end() const
        { return last; }

        int size() const
        { return std::distance(first, last); }
   };

    template<typename Iterator>
    Span <std::remove_reference_t<decltype(*std::declval<Iterator>())>,Iterator> 
    makeSpan(Iterator begin, Iterator end)
    {
        using T =  
        std::remove_reference_t<decltype(*std::declval<Iterator>())>;

        return Span<T, Iterator>{begin, end};
    }


}
