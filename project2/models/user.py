from dataclasses import dataclass, field
from typing import Optional, List
from abc import ABC, abstractmethod


@dataclass
class User:
    id: int
    name: str
    email: str


class UserRepository(ABC):
    """Abstract repository for user data operations."""

    @abstractmethod
    def get_all(self) -> List[User]:
        pass

    @abstractmethod
    def get_by_id(self, user_id: int) -> Optional[User]:
        pass

    @abstractmethod
    def create(self, name: str, email: str) -> User:
        pass

    @abstractmethod
    def update(self, user_id: int, name: str, email: str) -> Optional[User]:
        pass

    @abstractmethod
    def delete(self, user_id: int) -> bool:
        pass


class InMemoryUserRepository(UserRepository):
    """In-memory implementation of UserRepository."""

    def __init__(self):
        self._users: List[User] = []
        self._next_id: int = 1

    def get_all(self) -> List[User]:
        return self._users.copy()

    def get_by_id(self, user_id: int) -> Optional[User]:
        for user in self._users:
            if user.id == user_id:
                return user
        return None

    def create(self, name: str, email: str) -> User:
        user = User(id=self._next_id, name=name, email=email)
        self._next_id += 1
        self._users.append(user)
        return user

    def update(self, user_id: int, name: str, email: str) -> Optional[User]:
        for user in self._users:
            if user.id == user_id:
                user.name = name
                user.email = email
                return user
        return None

    def delete(self, user_id: int) -> bool:
        for i, user in enumerate(self._users):
            if user.id == user_id:
                self._users.pop(i)
                return True
        return False


# Global repository instance
user_repository = InMemoryUserRepository()
