#pragma once

namespace impl{
  
template <typename IntType>
class counting_iterator {
  static_assert(std::numeric_limits<IntType>::is_integer, "Cannot instantiate counting_iterator with a non-integer type");
public:
  using value_type = IntType;
  using difference_type = typename std::make_signed<IntType>::type;
  using pointer = IntType*;
  using reference = IntType&;
  using iterator_category = std::random_access_iterator_tag;

  counting_iterator() : value(0) { }
  explicit counting_iterator(IntType v) : value(v) { }

  value_type operator*() const { return value; }
  value_type operator[](difference_type n) const { return value + n; }

  counting_iterator& operator++() { ++value; return *this; }
  counting_iterator operator++(int) {
    counting_iterator result{value};
    ++value;
    return result;
  }
  counting_iterator& operator--() { --value; return *this; }
  counting_iterator operator--(int) {
    counting_iterator result{value};
    --value;
    return result;
  }
  counting_iterator& operator+=(difference_type n) { value += n; return *this; }
  counting_iterator& operator-=(difference_type n) { value -= n; return *this; }

  friend counting_iterator operator+(counting_iterator const& i, difference_type n)          { return counting_iterator(i.value + n);  }
  friend counting_iterator operator+(difference_type n, counting_iterator const& i)          { return counting_iterator(i.value + n);  }
  friend difference_type   operator-(counting_iterator const& x, counting_iterator const& y) { return x.value - y.value;  }
  friend counting_iterator operator-(counting_iterator const& i, difference_type n)          { return counting_iterator(i.value - n);  }

  friend bool operator==(counting_iterator const& x, counting_iterator const& y) { return x.value == y.value;  }
  friend bool operator!=(counting_iterator const& x, counting_iterator const& y) { return x.value != y.value;  }
  friend bool operator<(counting_iterator const& x, counting_iterator const& y)  { return x.value < y.value; }
  friend bool operator<=(counting_iterator const& x, counting_iterator const& y) { return x.value <= y.value; }
  friend bool operator>(counting_iterator const& x, counting_iterator const& y)  { return x.value > y.value; }
  friend bool operator>=(counting_iterator const& x, counting_iterator const& y) { return x.value >= y.value; }

private:
  IntType value;
};

} //impl
