# server.py

from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os
import time
import uuid

from master import DynamicMasterOrchestrator

app = Flask(__name__, static_folder='../', static_url_path='/')
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

app.secret_key = os.urandom(24)

# Store active conversations
active_conversations = {}


@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'frontend.html')


@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)


@app.route("/health")
def health():
    return jsonify({
        "ok": True, 
        "status": "Server running", 
        "time": time.time(),
        "active_conversations": len(active_conversations)
    })


@app.route("/chat", methods=["POST"])
def chat():
    """
    Conversational endpoint - processes ONE message at a time
    
    Request body:
    {
        "message": "User's message",
        "conversation_id": "optional - creates new if not provided",
        "userId": "optional - user email"
    }
    
    Response types:
    - "question": Agent needs more info
    - "complete": Analysis finished
    - "error": Something went wrong
    """
    start = time.time()
    
    try:
        payload = request.get_json(force=True)
        user_message = payload.get("message", "").strip()
        conversation_id = payload.get("conversation_id")
        user_id = payload.get("userId")
        
        if not user_message:
            return jsonify({
                "ok": False,
                "error": "No message provided"
            }), 400
        
        print(f"\nChat message received: {user_message[:80]}...")
        
        # Get or create conversation
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            print(f"Creating new conversation: {conversation_id}")
        
        # Get or create orchestrator for this conversation
        if conversation_id not in active_conversations:
            print(f"Creating new orchestrator for {conversation_id}")
            active_conversations[conversation_id] = DynamicMasterOrchestrator()
        
        orchestrator = active_conversations[conversation_id]
        
        # Process the message
        result = orchestrator.process_user_message(user_message)
        
        elapsed = int((time.time() - start) * 1000)
        
        # Build response based on result type
        response_type = result.get("type")
        
        if response_type == "question":
            return jsonify({
                "ok": True,
                "type": "question",
                "question": result.get("question"),
                "reason": result.get("reason"),
                "conversation_id": conversation_id,
                "conversation_history": result.get("conversation", []),
                "state": result.get("state", {}),
                "elapsed_ms": elapsed
            })
        
        elif response_type == "complete":
            final_result = result.get("result", {})
            # ADD THIS DEBUG:
            print(f"DEBUG - Agents executed: {final_result.get('agents_executed', [])}")
            # Debug: Check if references exist
            # Debug: Check if references exist
            print(f"DEBUG - Final result keys: {final_result.keys()}")
            print(f"DEBUG - Web results: {len(final_result.get('web_results', []))}")
            print(f"DEBUG - Web sources: {len(final_result.get('web_sources', []))}")
                      
            
            return jsonify({
                "ok": True,
                "type": "complete",
                "result": final_result,
                "conversation_id": conversation_id,
                "conversation_history": result.get("conversation", []),
                "execution_summary": result.get("execution_summary", {}),
                "elapsed_ms": elapsed
            })
        
        elif response_type == "error":
            return jsonify({
                "ok": False,
                "type": "error",
                "error": result.get("error", "Unknown error"),
                "conversation_id": conversation_id,
                "state": result.get("state", {}),
                "elapsed_ms": elapsed
            }), 500
        
        else:
            return jsonify({
                "ok": False,
                "error": f"Unknown response type: {response_type}"
            }), 500
    
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"Chat error: {err_msg}")
        return jsonify({
            "ok": False,
            "error": str(e),
            "trace": err_msg
        }), 500


@app.route("/chat/reset", methods=["POST"])
def reset_conversation():
    """
    Reset a conversation
    
    Request body:
    {
        "conversation_id": "session-id"
    }
    """
    try:
        payload = request.get_json(force=True)
        conversation_id = payload.get("conversation_id")
        
        if conversation_id and conversation_id in active_conversations:
            del active_conversations[conversation_id]
            print(f"Deleted conversation: {conversation_id}")
            
            return jsonify({
                "ok": True,
                "message": "Conversation reset successfully"
            })
        
        return jsonify({
            "ok": True,
            "message": "No active conversation to reset"
        })
    
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/admin/cleanup", methods=["POST"])
def cleanup_old_conversations():
    """Clean up old conversations"""
    try:
        cleaned = 0
        # TODO: Implement timestamp tracking and cleanup
        
        return jsonify({
            "ok": True,
            "message": "Cleanup completed",
            "conversations_cleaned": cleaned
        })
    
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    print("="*70)
    print("Starting Workout InjuryIntel Server")
    print("="*70)
    print("Endpoints:")
    print("   GET  /                  - Frontend UI")
    print("   POST /chat              - Conversational analysis")
    print("   POST /chat/reset        - Reset conversation")
    print("   GET  /health            - Health check")
    print("="*70)
    print("\nServer starting on http://0.0.0.0:5000\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
