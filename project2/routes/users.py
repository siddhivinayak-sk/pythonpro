from flask import Blueprint, jsonify, request
from models.user import user_repository

users_bp = Blueprint('users', __name__, url_prefix='/api/users')


@users_bp.route('', methods=['GET'])
def get_users():
    """Get all users."""
    users = user_repository.get_all()
    return jsonify([{'id': u.id, 'name': u.name, 'email': u.email} for u in users])


@users_bp.route('/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get a single user by ID."""
    user = user_repository.get_by_id(user_id)
    if user is None:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({'id': user.id, 'name': user.name, 'email': user.email})


@users_bp.route('', methods=['POST'])
def create_user():
    """Create a new user."""
    data = request.get_json()
    if not data or not data.get('name') or not data.get('email'):
        return jsonify({'error': 'Name and email are required'}), 400

    user = user_repository.create(name=data['name'], email=data['email'])
    return jsonify({'id': user.id, 'name': user.name, 'email': user.email}), 201


@users_bp.route('/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    """Update an existing user."""
    data = request.get_json()
    if not data or not data.get('name') or not data.get('email'):
        return jsonify({'error': 'Name and email are required'}), 400

    user = user_repository.update(user_id=user_id, name=data['name'], email=data['email'])
    if user is None:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({'id': user.id, 'name': user.name, 'email': user.email})


@users_bp.route('/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user."""
    success = user_repository.delete(user_id)
    if not success:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({'message': 'User deleted successfully'}), 200
