"""
WebSocket Manager for Real-time Updates
"""

import asyncio
import json
from typing import Dict, Set, Any, Optional
from datetime import datetime
import logging
from fastapi import WebSocket

from backend.models import WebSocketMessage

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and real-time updates"""
    
    def __init__(self):
        # Store active connections
        self.connections: Dict[str, WebSocket] = {}
        
        # Store subscriptions: client_id -> {resource_type -> {resource_id}}
        self.subscriptions: Dict[str, Dict[str, Set[str]]] = {}
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        async with self.lock:
            self.connections[client_id] = websocket
            self.subscriptions[client_id] = {}
        
        logger.info(f"WebSocket client {client_id} connected")
        
        # Send welcome message
        await self.send_to_client(client_id, {
            "type": "welcome",
            "client_id": client_id,
            "message": "Connected to Customer Support RL Environment API",
            "timestamp": datetime.now().isoformat()
        })
    
    async def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        async with self.lock:
            if client_id in self.connections:
                try:
                    await self.connections[client_id].close()
                except:
                    pass  # Connection might already be closed
                
                del self.connections[client_id]
                del self.subscriptions[client_id]
        
        logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_to_client(self, client_id: str, data: Dict[str, Any]):
        """Send data to a specific client"""
        async with self.lock:
            if client_id not in self.connections:
                return False
            
            try:
                message = WebSocketMessage(
                    type=data.get("type", "message"),
                    data=data
                )
                await self.connections[client_id].send_text(message.json())
                return True
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
                # Remove disconnected client
                await self.disconnect(client_id)
                return False
    
    async def broadcast(self, data: Dict[str, Any], resource_type: Optional[str] = None, resource_id: Optional[str] = None):
        """Broadcast data to all connected clients or subscribers"""
        async with self.lock:
            clients_to_notify = set()
            
            if resource_type and resource_id:
                # Send to subscribers of specific resource
                for client_id, subs in self.subscriptions.items():
                    if resource_type in subs and resource_id in subs[resource_type]:
                        clients_to_notify.add(client_id)
            else:
                # Send to all connected clients
                clients_to_notify = set(self.connections.keys())
            
            # Send to all relevant clients
            for client_id in clients_to_notify:
                await self.send_to_client(client_id, data)
    
    async def subscribe(self, client_id: str, resource_type: str, resource_id: str):
        """Subscribe client to updates for a specific resource"""
        async with self.lock:
            if client_id not in self.subscriptions:
                return False
            
            if resource_type not in self.subscriptions[client_id]:
                self.subscriptions[client_id][resource_type] = set()
            
            self.subscriptions[client_id][resource_type].add(resource_id)
            logger.info(f"Client {client_id} subscribed to {resource_type}:{resource_id}")
            
            # Send confirmation
            await self.send_to_client(client_id, {
                "type": "subscription_confirmed",
                "resource_type": resource_type,
                "resource_id": resource_id,
                "message": f"Subscribed to {resource_type}:{resource_id}"
            })
            
            return True
    
    async def unsubscribe(self, client_id: str, resource_type: str, resource_id: str):
        """Unsubscribe client from updates for a specific resource"""
        async with self.lock:
            if (client_id not in self.subscriptions or 
                resource_type not in self.subscriptions[client_id]):
                return False
            
            self.subscriptions[client_id][resource_type].discard(resource_id)
            
            # Clean up empty resource type
            if not self.subscriptions[client_id][resource_type]:
                del self.subscriptions[client_id][resource_type]
            
            logger.info(f"Client {client_id} unsubscribed from {resource_type}:{resource_id}")
            
            # Send confirmation
            await self.send_to_client(client_id, {
                "type": "unsubscription_confirmed",
                "resource_type": resource_type,
                "resource_id": resource_id,
                "message": f"Unsubscribed from {resource_type}:{resource_id}"
            })
            
            return True
    
    async def notify_environment_update(self, env_id: str, update_type: str, data: Dict[str, Any]):
        """Notify subscribers about environment updates"""
        message = {
            "type": "environment_update",
            "update_type": update_type,
            "environment_id": env_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast(message, "environment", env_id)
    
    async def notify_training_update(self, session_id: str, update_type: str, data: Dict[str, Any]):
        """Notify subscribers about training updates"""
        message = {
            "type": "training_update",
            "update_type": update_type,
            "session_id": session_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast(message, "training", session_id)
    
    async def notify_model_update(self, model_id: str, update_type: str, data: Dict[str, Any]):
        """Notify subscribers about model updates"""
        message = {
            "type": "model_update",
            "update_type": update_type,
            "model_id": model_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast(message, "model", model_id)
    
    async def send_system_alert(self, alert_type: str, message: str, level: str = "info"):
        """Send system-wide alert to all connected clients"""
        alert_data = {
            "type": "system_alert",
            "alert_type": alert_type,
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast(alert_data)
    
    async def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a connected client"""
        async with self.lock:
            if client_id not in self.connections:
                return None
            
            return {
                "client_id": client_id,
                "connected": True,
                "subscriptions": {
                    resource_type: list(resource_ids)
                    for resource_type, resource_ids in self.subscriptions[client_id].items()
                }
            }
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about WebSocket connections"""
        async with self.lock:
            total_connections = len(self.connections)
            total_subscriptions = sum(
                sum(len(resource_ids) for resource_ids in subs.values())
                for subs in self.subscriptions.values()
            )
            
            resource_type_counts = {}
            for subs in self.subscriptions.values():
                for resource_type in subs:
                    resource_type_counts[resource_type] = resource_type_counts.get(resource_type, 0) + 1
            
            return {
                "total_connections": total_connections,
                "total_subscriptions": total_subscriptions,
                "subscriptions_by_type": resource_type_counts,
                "connected_clients": list(self.connections.keys())
            }
    
    async def cleanup_disconnected_clients(self):
        """Clean up disconnected clients"""
        async with self.lock:
            disconnected_clients = []
            
            for client_id, websocket in self.connections.items():
                try:
                    # Try to send a ping to check if connection is alive
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except:
                    disconnected_clients.append(client_id)
            
            # Remove disconnected clients
            for client_id in disconnected_clients:
                await self.disconnect(client_id)
            
            if disconnected_clients:
                logger.info(f"Cleaned up {len(disconnected_clients)} disconnected clients")
    
    async def send_metrics_update(self, metrics_type: str, data: Dict[str, Any]):
        """Send metrics update to all subscribers"""
        message = {
            "type": "metrics_update",
            "metrics_type": metrics_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast(message)
    
    async def send_progress_update(self, resource_type: str, resource_id: str, progress: float, details: Optional[Dict[str, Any]] = None):
        """Send progress update for long-running operations"""
        message = {
            "type": "progress_update",
            "resource_type": resource_type,
            "resource_id": resource_id,
            "progress": progress,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast(message, resource_type, resource_id)
    
    async def handle_client_message(self, client_id: str, message: Dict[str, Any]):
        """Handle incoming message from client"""
        message_type = message.get("type")
        
        if message_type == "subscribe":
            resource_type = message.get("resource_type")
            resource_id = message.get("resource_id")
            if resource_type and resource_id:
                await self.subscribe(client_id, resource_type, resource_id)
        
        elif message_type == "unsubscribe":
            resource_type = message.get("resource_type")
            resource_id = message.get("resource_id")
            if resource_type and resource_id:
                await self.unsubscribe(client_id, resource_type, resource_id)
        
        elif message_type == "ping":
            await self.send_to_client(client_id, {"type": "pong"})
        
        elif message_type == "get_client_info":
            client_info = await self.get_client_info(client_id)
            await self.send_to_client(client_id, {
                "type": "client_info",
                "data": client_info
            })
        
        else:
            logger.warning(f"Unknown message type from client {client_id}: {message_type}")


# Global websocket manager instance
websocket_manager = WebSocketManager()
