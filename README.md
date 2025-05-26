# AI Multi-Agent Tutoring Bot

A sophisticated AI-powered tutoring system that leverages multiple specialized agents to provide comprehensive educational assistance across Mathematics, Physics, Chemistry, and Biology. The system features advanced memory management, vector embeddings, real-time monitoring, and intelligent query routing.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Components](#system-components)
- [Deployment Guide](#deployment-guide)
- [Configuration](#configuration)
- [Monitoring & Analytics](#monitoring--analytics)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Production Deployment](#production-deployment)

## Architecture Overview

The system is built using a microservices architecture with the following components:

- **Frontend**: Next.js React application with real-time chat interface
- **Backend**: FastAPI-based multi-agent system with specialized tutoring agents
- **Database**: Supabase (PostgreSQL) with vector embeddings for conversation memory
- **Cache**: Redis for session management and real-time data
- **Monitoring**: Prometheus metrics with Grafana visualization
- **Reverse Proxy**: Nginx for load balancing and SSL termination
- **Containerization**: Docker Compose for orchestration

## Features

### Multi-Agent System
- **MathAgent**: Advanced equation solving, calculus, algebra with SymPy integration
- **PhysicsAgent**: Physics constants, kinematics calculations, and numerical simulations
- **ChemistryAgent**: Molecular property analysis, equation balancing with RDKit
- **BiologyAgent**: DNA sequence analysis, protein translation, and biological computations
- **TutorAgent**: Main orchestrator with intelligent query classification

### Advanced Memory Management
- **Vector Embeddings**: Conversation context stored using OpenAI embeddings
- **Semantic Search**: Retrieve relevant past conversations using vector similarity
- **Session Persistence**: Redis-based session management with configurable timeout
- **Conversation Summaries**: AI-generated summaries of tutoring sessions

### Real-time Features
- **Live Chat Interface**: WebSocket-enabled real-time communication
- **Typing Indicators**: Visual feedback during AI processing
- **Session Management**: Persistent chat history across browser sessions
- **Multi-tab Support**: Synchronized state across multiple browser tabs

### Monitoring & Analytics
- **Prometheus Metrics**: Request rates, response times, tool usage tracking
- **Grafana Dashboard**: Real-time visualization of system performance
- **Health Checks**: Comprehensive service health monitoring
- **Error Tracking**: Detailed logging and error reporting

## Technology Stack

### Frontend
- **Next.js 14**: React framework with server-side rendering
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Shadcn/UI**: Modern component library
- **Recharts**: Data visualization
- **React Markdown**: Mathematical notation rendering with KaTeX

### Backend
- **FastAPI**: High-performance Python web framework
- **Google Gemini**: Advanced language model for AI responses
- **OpenAI**: Embeddings for vector search
- **Supabase**: PostgreSQL database with vector extensions
- **Redis**: In-memory data structure store
- **LangChain**: AI application framework
- **Scientific Libraries**: SymPy, SciPy, RDKit, ChemPy

### Infrastructure
- **Docker**: Containerization platform
- **Nginx**: High-performance web server and reverse proxy
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Metrics visualization and alerting
- **Let's Encrypt**: SSL certificate management

## System Components

### Frontend Application

The React-based frontend provides an intuitive chat interface with:

- **Responsive Design**: Optimized for desktop and mobile devices
- **Real-time Chat**: WebSocket connection for instant messaging
- **Agent Visualization**: Visual indicators for which AI agent is responding
- **Tool Usage Display**: Shows which computational tools were used
- **Analytics Dashboard**: User engagement and conversation metrics
- **Example Queries**: Pre-built examples for each subject area

### Backend Services

#### Multi-Agent Architecture

The backend implements a sophisticated agent system:

```python
# Agent Specialization
- MathAgent: Equation solving, calculus, algebra
- PhysicsAgent: Constants lookup, kinematics, simulations  
- ChemistryAgent: Molecular analysis, equation balancing
- BiologyAgent: DNA analysis, protein translation
- TutorAgent: Query classification and orchestration
```

#### Memory Management System

Advanced conversation memory using vector embeddings:

```python
# Vector Storage
- OpenAI text-embedding-3-small for semantic encoding
- Supabase vector extensions for similarity search
- Redis for session caching and real-time access
- Automated conversation summarization
```

#### Tool Integration

Each agent includes specialized computational tools:

- **Mathematical Tools**: SymPy for symbolic computation
- **Physics Tools**: SciPy constants and numerical methods
- **Chemistry Tools**: RDKit for molecular property calculation
- **Biology Tools**: DNA sequence analysis and protein translation

### Database Schema

#### Supabase Tables

**Conversations Table**
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMPTZ DEFAULT now()
);
```

**Vector Embeddings Table**
```sql
CREATE TABLE conversation_vectors (
    id UUID PRIMARY KEY REFERENCES conversations(id),
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB
);
```

### Redis Schema

Session management and caching:
```
session:{session_id}:history - Recent conversation history
session:{session_id}:context - Conversation context cache
agent:{agent_name}:stats - Agent performance metrics
```

## Deployment Guide

### Prerequisites

- Digital Ocean Droplet (minimum 4GB RAM, 2 CPU cores)
- Docker and Docker Compose installed
- Domain name with DNS configured
- Required API keys (see Configuration section)

### Digital Ocean Setup

1. **Create Droplet**
```bash
# Create a new droplet with Ubuntu 22.04 LTS
# Minimum specs: 4GB RAM, 2 CPUs, 80GB SSD
```

2. **Initial Server Setup**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y curl wget git ufw

# Configure firewall
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw enable
```

3. **Install Docker**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Application Deployment

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/tutorbot-system.git
cd tutorbot-system
```

2. **Environment Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

3. **Environment Variables**
```bash
# AI Service Keys
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key

# Database Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Application Settings
REDIS_URL=redis://redis:6379
EMBEDDING_MODEL=text-embedding-3-small
SESSION_TIMEOUT=3600
MAX_MEMORY_ITEMS=50

# Monitoring
PROMETHEUS_PORT=9090
```

4. **SSL Certificate Setup**
```bash
# Initialize SSL certificates with Certbot
sudo docker-compose exec certbot certbot certonly --webroot --webroot-path=/var/www/certbot -d your-domain.com

# Set up automatic renewal
echo "0 12 * * * /usr/local/bin/docker-compose -f /path/to/docker-compose.yml exec certbot certbot renew --quiet" | sudo crontab -
```

5. **Start Services**
```bash
# Build and start all services
docker-compose up --build -d

# Verify services are running
docker-compose ps

# Check service logs
docker-compose logs -f backend
```

### Service Verification

1. **Frontend**: http://your-domain.com
2. **Backend API**: http://your-domain.com/api/health
3. **Grafana Dashboard**: http://your-domain.com/grafana
4. **Prometheus Metrics**: http://your-domain.com:9090

## Configuration

### Nginx Configuration

The Nginx configuration provides:

- **Reverse Proxy**: Routes requests to appropriate services
- **SSL Termination**: Handles HTTPS certificates
- **Rate Limiting**: Protects against abuse
- **Static File Serving**: Optimized delivery of frontend assets
- **WebSocket Support**: Enables real-time chat functionality

Key configuration features:
```nginx
# API rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

# CORS headers for API access
add_header Access-Control-Allow-Origin "*" always;

# WebSocket upgrade handling
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
```

### Docker Compose Architecture

The Docker Compose setup orchestrates:

- **Multi-service networking** with isolated container communication
- **Volume persistence** for data storage and SSL certificates
- **Health checks** for service dependency management
- **Environment variable injection** for configuration management
- **Resource limits** to prevent resource exhaustion

## Monitoring & Analytics

### Grafana Dashboard

The system includes a comprehensive Grafana dashboard accessible at `/grafana` with the following panels:

#### Request Analytics
- **Request Rate by Agent**: Real-time visualization of agent utilization
- **Active Sessions**: Current number of concurrent users
- **Requests by Subject**: Distribution of queries across academic subjects

#### Performance Metrics
- **Response Time Percentiles**: 95th percentile and median response times
- **Tool Usage Rate**: Frequency of computational tool utilization
- **Error Rate Tracking**: System reliability metrics

#### System Health
- **Service Status**: Health checks for all components
- **Resource Utilization**: CPU and memory usage monitoring
- **Database Performance**: Query performance and connection pooling

### Prometheus Metrics

Custom metrics tracked:
```python
# Request counting by agent and subject
tutorbot_requests_total{agent="MathAgent", subject="math"}

# Response time histograms
tutorbot_response_time_seconds_bucket{agent="PhysicsAgent"}

# Active session tracking
tutorbot_active_sessions

# Tool usage monitoring
tutorbot_tool_usage_total{tool="equation_solver", agent="MathAgent"}
```

### Chat History & Memory

The system implements sophisticated conversation memory:

#### Vector Embedding Storage
- **Semantic Search**: Retrieve contextually relevant past conversations
- **Long-term Memory**: Persistent storage of all user interactions
- **Privacy Controls**: Session-based data isolation

#### Conversation Context
- **Smart Context Retrieval**: AI selects most relevant past discussions
- **Memory Summarization**: Automatic generation of conversation summaries
- **Context-Aware Responses**: Agents reference previous interactions

#### Session Management
```python
# Redis-based session storage
session_data = {
    "session_id": "uuid-string",
    "created_at": "timestamp",
    "last_activity": "timestamp",
    "conversation_count": 42,
    "subjects_covered": ["math", "physics"],
    "context_summary": "Generated AI summary"
}
```

## API Documentation

### Core Endpoints

#### Query Processing
```http
POST /api/ask
Content-Type: application/json

{
    "query": "Solve the equation 2x + 5 = 11",
    "session_id": "optional-session-id"
}
```

#### Conversation History
```http
GET /api/conversation/history/{session_id}?limit=10

Response:
{
    "session_id": "uuid",
    "history": [
        {
            "query": "User question",
            "response": "AI response",
            "agent_used": "MathAgent",
            "tools_used": ["equation_solver"],
            "timestamp": "ISO-8601"
        }
    ]
}
```

#### Session Summary
```http
GET /api/conversation/summary/{session_id}

Response:
{
    "session_id": "uuid",
    "summary": "AI-generated conversation summary",
    "total_interactions": 15,
    "topics_covered": ["algebra", "calculus"]
}
```

### Health and Monitoring
```http
GET /api/health
GET /api/metrics  # Prometheus format
GET /api/agents   # Available agent information
```

## Development

### Local Development Setup

1. **Clone and Setup**
```bash
git clone https://github.com/yourusername/tutorbot-system.git
cd tutorbot-system
```

2. **Environment Setup**
```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup  
cd ../frontend
npm install
```

3. **Local Services**
```bash
# Start Redis locally
docker run -d --name redis -p 6379:6379 redis:alpine

# Start development servers
cd backend && uvicorn main:app --reload
cd frontend && npm run dev
```

### Testing

```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend  
npm run test

# Integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## Production Deployment

For production deployment on Digital Ocean:

### Domain Configuration
1. Configure DNS A record pointing to your droplet IP
2. Update Nginx configuration with your domain
3. Generate SSL certificates using Let's Encrypt

### Performance Optimization
- Enable Gzip compression in Nginx
- Configure Redis persistence
- Set up database connection pooling
- Implement CDN for static assets

### Security Hardening
- Configure UFW firewall rules
- Enable fail2ban for intrusion prevention
- Regular security updates via automated scripts
- API rate limiting and request validation

### Backup Strategy
```bash
# Database backups
pg_dump supabase_db > backup-$(date +%Y%m%d).sql

# Application data backup
docker run --rm -v tutorbot_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis-backup.tar.gz /data
```

### Scaling Considerations
- Horizontal scaling with Docker Swarm or Kubernetes
- Database read replicas for improved performance
- Redis Cluster for high availability
- Load balancer configuration for multiple instances

## Live Demo

**Frontend Application**: [http://138.197.72.190.nip.io/](http://138.197.72.190.nip.io/)

**Grafana Dashboard**: [http://138.197.72.190.nip.io:3001/](http://138.197.72.190.nip.io:3001/)
- Username: admin
- Password: admin

The live demo showcases:
- Real-time multi-agent conversations
- Advanced mathematical computations
- Scientific tool integrations
- Comprehensive monitoring dashboards
- Session persistence and memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request with detailed description

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For technical support or questions:
- Create an issue in the GitHub repository
- Check the documentation wiki
- Review the API documentation at `/docs` endpoint
