'use client'


import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Brain, Beaker, Dna, Calculator, RefreshCw, Menu, X, ChevronRight, Sparkles, Clock, BookOpen, BarChart3 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { Components } from 'react-markdown';

interface CodeBlockProps {
  node?: any;
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  agent?: string;
  subject?: string;
  tools_used?: string[];
}

interface Analytics {
  totalMessages: number;
  userQuestions: number;
  subjectsData: Array<{ name: string; value: number }>;
}

const renderMarkdown = (content: string) => {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        code: ({ node, inline, className, children, ...props }: CodeBlockProps) => {
          const match = /language-(\w+)/.exec(className || '');
          return !inline && match ? (
            <SyntaxHighlighter
              style={vscDarkPlus}
              language={match[1]}
              PreTag="div"
              {...props}
            >
              {String(children).replace(/\n$/, '')}
            </SyntaxHighlighter>
          ) : (
            <code className={className} {...props}>
              {children}
            </code>
          );
        }
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

export default function AITutorChat() {
  const [sessionId, setSessionId] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [systemStatus, setSystemStatus] = useState<Record<string, string>>({});
  const [activeTab, setActiveTab] = useState('chat');
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initialize session
    const id = generateSessionId();
    setSessionId(id);
    checkSystemStatus();
    
    // Load chat history from localStorage
    const savedMessages = localStorage.getItem(`chat-history-${id}`);
    if (savedMessages) {
      setMessages(JSON.parse(savedMessages));
    }
  }, []);

  useEffect(() => {
    // Save messages to localStorage
    if (sessionId && messages.length > 0) {
      localStorage.setItem(`chat-history-${sessionId}`, JSON.stringify(messages));
    }
  }, [messages, sessionId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const generateSessionId = () => {
    return 'xxxx-xxxx-xxxx'.replace(/[x]/g, () => {
      return (Math.random() * 16 | 0).toString(16);
    });
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const checkSystemStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      if (response.ok) {
        const data = await response.json();
        setSystemStatus(data.services || {});
      }
    } catch (error) {
      console.error('Failed to check system status:', error);
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: input,
          session_id: sessionId
        })
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage: Message = {
          id: Date.now() + 1,
          role: 'assistant',
          content: data.response,
          agent: data.agent_used,
          subject: data.subject,
          tools_used: data.tools_used || [],
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, botMessage]);
        updateAnalytics();
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      const errorMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const updateAnalytics = () => {
    const subjects = messages.filter(m => m.role === 'assistant' && m.subject)
      .map(m => m.subject);
    
    const subjectCounts: Record<string, number> = subjects.reduce((acc, subject) => {
      if (subject) {
        acc[subject] = (acc[subject] || 0) + 1;
      }
      return acc;
    }, {} as Record<string, number>);

    setAnalytics({
      totalMessages: messages.length,
      userQuestions: messages.filter(m => m.role === 'user').length,
      subjectsData: Object.entries(subjectCounts).map(([name, value]) => ({ name, value }))
    });
  };

  const startNewSession = () => {
    const newId = generateSessionId();
    setSessionId(newId);
    setMessages([]);
    setAnalytics(null);
  };

  const examples = {
    Mathematics: [
      "Can you help me solve 2x + 5 = 11?",
      "What is the derivative of x³ + 2x² - 5x + 3?",
      "Calculate the integral of sin(x) * cos(x)",
      "Explain the quadratic formula with an example"
    ],
    Physics: [
      "What is Newton's second law?",
      "What is the speed of light?",
      "Calculate velocity after 5 seconds",
      "Explain gravitational force"
    ],
    Chemistry: [
      "What are the properties of H2O?",
      "Calculate molecular weight of glucose",
      "Explain the concept of pH",
      "What is a covalent bond?"
    ],
    Biology: [
      "Analyze DNA sequence ATCGATCG",
      "What is photosynthesis?",
      "Explain cell structure",
      "Role of mitochondria?"
    ]
  };

  const agents = [
    {
      name: "MathAgent",
      icon: Calculator,
      color: "text-blue-500",
      description: "Solves mathematical problems and equations"
    },
    {
      name: "PhysicsAgent",
      icon: Brain,
      color: "text-purple-500",
      description: "Explains physics concepts and calculations"
    },
    {
      name: "ChemistryAgent",
      icon: Beaker,
      color: "text-green-500",
      description: "Handles chemistry questions and molecular analysis"
    },
    {
      name: "BiologyAgent",
      icon: Dna,
      color: "text-pink-500",
      description: "Explores biological concepts and DNA analysis"
    }
  ];

  const getAgentIcon = (agentName: string) => {
    const agent = agents.find(a => a.name === agentName);
    if (!agent) return Bot;
    return agent.icon;
  };

  const getAgentColor = (agentName: string) => {
    const agent = agents.find(a => a.name === agentName);
    return agent?.color || "text-gray-500";
  };

  const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#ec4899'];

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 overflow-hidden`}>
        <div className="p-6 h-full flex flex-col">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
              AI Tutor
            </h2>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(false)}
              className="md:hidden"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          <div className="space-y-4 mb-6">
            <Alert>
              <Sparkles className="h-4 w-4" />
              <AlertTitle>Session ID</AlertTitle>
              <AlertDescription className="font-mono text-xs">
                {sessionId}
              </AlertDescription>
            </Alert>

            <Button 
              onClick={startNewSession}
              className="w-full"
              variant="outline"
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              New Session
            </Button>
          </div>

          <Separator className="mb-6" />

          <div className="space-y-4 flex-1">
            <div>
              <h3 className="text-sm font-semibold mb-2">System Status</h3>
              <div className="space-y-2">
                {Object.entries(systemStatus).map(([service, status]) => (
                  <div key={service} className="flex items-center justify-between">
                    <span className="text-sm capitalize">{service}</span>
                    <Badge variant={status === 'connected' ? 'default' : 'secondary'}>
                      {status}
                    </Badge>
                  </div>
                ))}
              </div>
            </div>

            <Separator />

            <div className="space-y-2">
              <h3 className="text-sm font-semibold mb-2">Recent Topics</h3>
              {[...new Set(messages.filter(m => m.subject).map(m => m.subject))].slice(-5).map((topic, idx) => (
                <Badge key={idx} variant="outline" className="mr-2">
                  {topic}
                </Badge>
              ))}
            </div>
          </div>

          <div className="mt-auto pt-4">
            <p className="text-xs text-gray-500 text-center">
              Multi-Agent Tutoring System
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSidebarOpen(!sidebarOpen)}
              >
                <Menu className="h-5 w-5" />
              </Button>
              <h1 className="text-xl font-semibold">AI Multi-Agent Tutor</h1>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline">
                <Clock className="mr-1 h-3 w-3" />
                {messages.length} messages
              </Badge>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
          <TabsList className="grid w-full grid-cols-4 max-w-2xl mx-auto mt-4">
            <TabsTrigger value="chat">Chat</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="agents">Agents</TabsTrigger>
            <TabsTrigger value="examples">Examples</TabsTrigger>
          </TabsList>

          <TabsContent value="chat" className="flex-1 flex flex-col p-4">
            <Card className="flex-1 flex flex-col max-w-4xl mx-auto w-full">
              <CardContent className="flex-1 p-0">
                <ScrollArea className="h-[calc(100vh-300px)] p-6">
                  {messages.length === 0 ? (
                    <div className="text-center py-12">
                      <Bot className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                      <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                        Start a conversation
                      </h3>
                      <p className="text-gray-500 mt-2">
                        Ask me anything about Math, Physics, Chemistry, or Biology!
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {messages.map((message) => (
                        <div
                          key={message.id}
                          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div className={`flex space-x-3 max-w-[80%] ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                            <Avatar className="h-8 w-8">
                              <AvatarFallback className={message.role === 'user' ? 'bg-blue-500' : 'bg-gray-500'}>
                                {message.role === 'user' ? (
                                  <User className="h-4 w-4 text-white" />
                                ) : (
                                  message.agent && React.createElement(getAgentIcon(message.agent), {
                                    className: "h-4 w-4 text-white"
                                  })
                                )}
                              </AvatarFallback>
                            </Avatar>
                            <div className="space-y-2">
                              <div className={`rounded-lg p-4 ${
                                message.role === 'user' 
                                  ? 'bg-blue-500 text-white' 
                                  : 'bg-gray-100 dark:bg-gray-800'
                              }`}>
                                {message.role === 'assistant' && (
                                  <div className="flex items-center space-x-2 mb-2">
                                    {message.agent && (
                                      <span className={`text-sm font-medium ${getAgentColor(message.agent)}`}>
                                        {message.agent}
                                      </span>
                                    )}
                                    {message.subject && (
                                      <Badge variant="outline" className="text-xs">
                                        {message.subject}
                                      </Badge>
                                    )}
                                  </div>
                                )}
                                <div className="prose dark:prose-invert max-w-none">
                                  {renderMarkdown(message.content)}
                                </div>
                                {message.tools_used && message.tools_used.length > 0 && (
                                  <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                                    <p className="text-xs text-gray-500">
                                      Tools used: {message.tools_used.join(', ')}
                                    </p>
                                  </div>
                                )}
                              </div>
                              <p className="text-xs text-gray-500">
                                {new Date(message.timestamp).toLocaleTimeString()}
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                      <div ref={messagesEndRef} />
                    </div>
                  )}
                </ScrollArea>
              </CardContent>
              <CardFooter className="p-4 border-t">
                <div className="flex space-x-2 w-full">
                  <Input
                    placeholder="Ask a question about Math, Physics, Chemistry, or Biology..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey && !loading) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    disabled={loading}
                    className="flex-1 min-h-[40px]"
                    style={{ resize: 'vertical', minHeight: '40px', maxHeight: '200px' }}
                  />
                  <Button onClick={sendMessage} disabled={loading || !input.trim()}>
                    {loading ? (
                      <RefreshCw className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="analytics" className="p-4">
            <div className="max-w-6xl mx-auto space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium">Total Messages</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{messages.length}</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium">Your Questions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {messages.filter(m => m.role === 'user').length}
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium">Subjects Covered</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {analytics?.subjectsData?.length || 0}
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {[...new Set(messages.filter(m => m.agent).map(m => m.agent))].length}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {analytics?.subjectsData && analytics.subjectsData.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Subject Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={analytics.subjectsData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {analytics.subjectsData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              <Card>
                <CardHeader>
                  <CardTitle>Response Time by Agent</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={[
                      { name: 'MathAgent', time: 0.8 },
                      { name: 'PhysicsAgent', time: 1.2 },
                      { name: 'ChemistryAgent', time: 0.9 },
                      { name: 'BiologyAgent', time: 1.1 }
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="time" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="agents" className="p-4">
            <div className="max-w-4xl mx-auto">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {agents.map((agent) => (
                  <Card key={agent.name} className="hover:shadow-lg transition-shadow">
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-3">
                        <div className={`p-2 rounded-lg bg-gray-100 dark:bg-gray-800 ${agent.color}`}>
                          {React.createElement(agent.icon, { className: "h-5 w-5" })}
                        </div>
                        <span>{agent.name}</span>
                      </CardTitle>
                      <CardDescription>{agent.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div>
                          <p className="text-sm font-medium mb-1">Capabilities:</p>
                          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            <li>• Problem solving</li>
                            <li>• Concept explanation</li>
                            <li>• Step-by-step guidance</li>
                            <li>• Visual representations</li>
                          </ul>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="examples" className="p-4 overflow-auto">
            <div className="max-w-4xl mx-auto space-y-6">
              {Object.entries(examples).map(([subject, questions]) => (
                <Card key={subject} className="overflow-hidden">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <BookOpen className="h-5 w-5" />
                      <span>{subject}</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {questions.map((question, idx) => (
                        <Button
                          key={idx}
                          variant="outline"
                          className="justify-start text-left h-auto py-3 px-4 whitespace-normal break-words"
                          onClick={() => {
                            setInput(question);
                            setActiveTab('chat');
                          }}
                        >
                          <ChevronRight className="h-4 w-4 mr-2 flex-shrink-0" />
                          <span className="text-sm">{question}</span>
                        </Button>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}