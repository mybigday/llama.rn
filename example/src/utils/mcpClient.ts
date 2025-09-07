// eslint-disable-next-line import/no-unresolved
import { Client } from '@modelcontextprotocol/sdk/client'
import { URL } from 'whatwg-url-without-unicode'
import {
  SSEJSStreamableHTTPClientTransport,
  SSEJSClientTransport,
} from 'mcp-sdk-client-ssejs'

import type { MCPConfig, MCPServer } from './storage'

export interface MCPTool {
  name: string
  description: string
  inputSchema: any
}

export interface MCPConnection {
  serverId: string
  connected: boolean
  error?: string
  tools: MCPTool[]
  client?: Client
}

export class MCPClientManager {
  private connections: Map<string, MCPConnection> = new Map()

  private config: MCPConfig = { mcpServers: {} }

  updateConfig(config: MCPConfig) {
    this.config = config
  }

  async connectToServers(): Promise<void> {
    await this.disconnect()
    const promises = Object.entries(this.config.mcpServers).map(
      ([serverId, server]) => this.connectToServer(serverId, server),
    )

    await Promise.allSettled(promises)
  }

  private async connectToServer(
    serverId: string,
    server: MCPServer,
  ): Promise<void> {
    try {
      // Initialize connection status
      this.connections.set(serverId, {
        serverId,
        connected: false,
        tools: [],
      })

      // For now, we'll simulate connection and tool discovery
      // In a real implementation, you would use the MCP SDK to connect
      if (server.type === 'streamable-http') {
        await this.connectStreamableHttp(serverId, server)
      } else if (server.type === 'sse') {
        await this.connectSSE(serverId, server)
      }
    } catch (error: any) {
      console.error('Error connecting to MCP server:', error)
      this.connections.set(serverId, {
        serverId,
        connected: false,
        error: error.message,
        tools: [],
      })
    }
  }

  private async connectStreamableHttp(
    serverId: string,
    server: MCPServer,
  ): Promise<void> {
    try {
      // Implementation for streamable-http MCP connection
      // This would use the actual MCP SDK when available
      const client = await MCPClientManager.createMCPClient(server)

      // List available tools
      const toolsResponse = await client.listTools()
      const tools: MCPTool[] = toolsResponse.tools.map((tool: any) => ({
        name: tool.name,
        description: tool.description || 'No description provided',
        inputSchema: tool.inputSchema,
      }))

      this.connections.set(serverId, {
        serverId,
        connected: true,
        tools,
        client,
      })
    } catch (error: any) {
      this.connections.set(serverId, {
        serverId,
        connected: false,
        error: `HTTP connection failed: ${error.message}`,
        tools: [],
      })
    }
  }

  private async connectSSE(serverId: string, server: MCPServer): Promise<void> {
    try {
      // Implementation for SSE MCP connection
      // This would use the actual MCP SDK when available
      const client = await MCPClientManager.createMCPClient(server)

      // List available tools
      const toolsResponse = await client.listTools()
      const tools: MCPTool[] = toolsResponse.tools.map((tool: any) => ({
        name: tool.name,
        description: tool.description || 'No description provided',
        inputSchema: tool.inputSchema,
      }))

      this.connections.set(serverId, {
        serverId,
        connected: true,
        tools,
        client,
      })
    } catch (error: any) {
      this.connections.set(serverId, {
        serverId,
        connected: false,
        error: `SSE connection failed: ${error.message}`,
        tools: [],
      })
    }
  }

  private static async createMCPClient(server: MCPServer): Promise<Client> {
    // Create transport based on server type
    const transport =
      server.type === 'sse'
        ? new SSEJSClientTransport(new URL(server.url), {
            eventSourceInit: { headers: server.headers || {} },
            requestInit: { headers: server.headers || {} },
            URL,
          })
        : new SSEJSStreamableHTTPClientTransport(new URL(server.url), {
            eventSourceInit: { headers: server.headers || {} },
            requestInit: { headers: server.headers || {} },
            URL,
          })

    // Create MCP client
    const client = new Client(
      {
        name: 'llama-rn-client',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      },
    )

    client.onerror = (err: any) => {
      console.error('Error connecting to MCP server:', err)
    }

    // Connect to the server
    await client.connect(transport)
    return client
  }

  getConnections(): MCPConnection[] {
    return Array.from(this.connections.values())
  }

  getConnection(serverId: string): MCPConnection | undefined {
    return this.connections.get(serverId)
  }

  getAllTools(): MCPTool[] {
    const allTools: MCPTool[] = []
    this.connections.forEach((connection) => {
      if (connection.connected) {
        allTools.push(...connection.tools)
      }
    })
    return allTools
  }

  async executeTool(toolName: string, args: any): Promise<any> {
    // Find which connection has this tool
    const connection = Array.from(this.connections.values()).find(
      (conn) =>
        conn.connected &&
        conn.client &&
        conn.tools.some((tool) => tool.name === toolName),
    )

    if (!connection || !connection.client) {
      throw new Error(`Tool ${toolName} not found in any connected MCP server`)
    }

    try {
      const result = await connection.client.callTool({
        name: toolName,
        arguments: args,
      })

      // Handle different response formats from MCP
      if (result.content) {
        return Array.isArray(result.content)
          ? result.content.map((c: any) => c.text || c).join('\n')
          : result.content
      }

      return result
    } catch (error: any) {
      throw new Error(`Tool execution failed: ${error.message}`)
    }
  }

  async disconnect(): Promise<void> {
    // Close all client connections
    const closePromises = Array.from(this.connections.values()).map(
      async (connection) => {
        if (connection.client) {
          try {
            await connection.client.close()
          } catch (error) {
            console.error('Error closing MCP client:', error)
          }
        }
      },
    )

    await Promise.all(closePromises)
    this.connections.clear()
  }

  async disconnectServer(serverId: string): Promise<void> {
    const connection = this.connections.get(serverId)
    if (connection?.client) {
      try {
        await connection.client.close()
      } catch (error) {
        console.error(`Error closing MCP client for ${serverId}:`, error)
      }
    }
    this.connections.delete(serverId)
  }
}

// Global instance
export const mcpClientManager = new MCPClientManager()
