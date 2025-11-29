package mcp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

const (
	ProviderGemini = "gemini"
	// DefaultGeminiBaseURL = "https://generativelanguage.googleapis.com/v1beta/models" // Example
)

type GeminiClient struct {
	*Client
}

// NewGeminiClientWithOptions 创建 Gemini 客户端
func NewGeminiClientWithOptions(opts ...ClientOption) AIClient {
	// 1. 创建预设选项
	geminiOpts := []ClientOption{
		WithProvider(ProviderGemini),
	}

	// 2. 合并用户选项
	allOpts := append(geminiOpts, opts...)

	// 3. 创建基础客户端
	baseClient := NewClient(allOpts...).(*Client)

	// 4. 创建 Gemini 客户端
	gClient := &GeminiClient{
		Client: baseClient,
	}

	// 5. 设置 hooks 指向 GeminiClient
	baseClient.hooks = gClient

	return gClient
}

func (c *GeminiClient) buildUrl() string {
	if c.UseFullURL {
		return c.BaseURL
	}

	// 如果 URL 已经包含 :generateContent，直接返回
	if strings.HasSuffix(c.BaseURL, ":generateContent") {
		return c.BaseURL
	}

	// 如果 BaseURL 指向具体的模型（例如 .../models/gemini-1.5-pro），直接附加
	if strings.Contains(c.BaseURL, "/models/") || strings.Contains(c.BaseURL, "/publishers/google/models/") {
		return fmt.Sprintf("%s:generateContent", c.BaseURL)
	}

	// 否则假设 BaseURL 是根路径，需要拼接模型名称
	// e.g. https://generativelanguage.googleapis.com/v1beta
	// -> https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent
	url := c.BaseURL
	if strings.HasSuffix(url, "/") {
		url = strings.TrimSuffix(url, "/")
	}
	return fmt.Sprintf("%s/models/%s:generateContent", url, c.Model)
}

func (c *GeminiClient) setAuthHeader(reqHeader http.Header) {
	// 如果是 Vertex AI (有 Service Account)，使用父类逻辑 (OAuth2 Token)
	if c.ServiceAccountJSON != "" && strings.Contains(c.BaseURL, "aiplatform.googleapis.com") {
		c.Client.setAuthHeader(reqHeader)
		return
	}

	// 如果是 AI Studio (generativelanguage.googleapis.com)，使用 x-goog-api-key
	// 或者用户没有提供 Service Account，默认认为是 API Key 模式
	reqHeader.Set("x-goog-api-key", c.APIKey)
}

func (c *GeminiClient) buildMCPRequestBody(systemPrompt, userPrompt string) map[string]any {
	// 构建 contents
	contents := []map[string]interface{}{
		{
			"role": "user",
			"parts": []map[string]interface{}{
				{"text": userPrompt},
			},
		},
	}

	requestBody := map[string]interface{}{
		"contents": contents,
		"generationConfig": map[string]interface{}{
			"temperature":     c.config.Temperature,
			"maxOutputTokens": c.MaxTokens,
		},
	}

	// System Instruction
	if systemPrompt != "" {
		requestBody["systemInstruction"] = map[string]interface{}{
			"parts": []map[string]interface{}{
				{"text": systemPrompt},
			},
		}
	}

	// Thinking Level
	if c.ThinkingLevel != "" {
		// 转换为大写 (LOW, HIGH)
		level := strings.ToUpper(c.ThinkingLevel)
		genConfig := requestBody["generationConfig"].(map[string]interface{})
		genConfig["thinkingLevel"] = level
	}

	return requestBody
}

func (c *GeminiClient) parseMCPResponse(body []byte) (string, error) {
	var result struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		Error struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
			Status  string `json:"status"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("解析响应失败: %w", err)
	}

	if result.Error.Code != 0 || result.Error.Message != "" {
		return "", fmt.Errorf("API返回错误: %s (Code: %d, Status: %s)", result.Error.Message, result.Error.Code, result.Error.Status)
	}

	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("API返回空响应 (Candidates empty)")
	}

	return result.Candidates[0].Content.Parts[0].Text, nil
}
