<script lang="ts">
    import AudioPlayer from '$lib/components/AudioPlayer.svelte';
    import DebugInfo from '$lib/components/DebugInfo.svelte';
    import LoadingSpinner from '$lib/components/LoadingSpinner.svelte';
    import type { AudioGenerationResponse } from '$lib/elevenlabs/client';

    let text = '';
    let voiceId = '';
    let loading = false;
    let result: AudioGenerationResponse | null = null;

    async function generateAudio() {
        if (!text) return;
        
        loading = true;
        result = null;
        
        try {
            const response = await fetch('/api/tts', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text,
                    voice_id: voiceId || undefined,
                    type: 'simple'
                })
            });
            
            const data = await response.json() as AudioGenerationResponse;
            
            if (!data.success) {
                throw new Error(data.message);
            }
            
            result = data;
        } catch (error) {
            result = {
                success: false,
                message: error instanceof Error ? error.message : String(error),
                debugInfo: []
            };
        } finally {
            loading = false;
        }
    }
</script>

<main>
    <h2>Basic Text-to-Speech Conversion</h2>
    <p class="page-description">Convert single text input to speech using optional voice ID.</p>
    
    <form on:submit|preventDefault={generateAudio} class="tts-form">
        <div class="form-group">
            <label for="text">Text</label>
            <textarea
                id="text"
                bind:value={text}
                placeholder="Enter text to convert to speech..."
                rows="4"
                required
            ></textarea>
        </div>
        
        <div class="form-group">
            <label for="voice">Voice ID (optional)</label>
            <input
                id="voice"
                type="text"
                bind:value={voiceId}
                placeholder="Enter voice ID..."
            />
        </div>
        
        <button type="submit" disabled={loading || !text}>
            {#if loading}
                <LoadingSpinner size={16} />
                Generating...
            {:else}
                Generate Audio
            {/if}
        </button>
    </form>
    
    {#if result}
        <div class="result">
            {#if result.success && result.audioData}
                <AudioPlayer 
                    audioData={result.audioData.data}
                    name={result.audioData.name}
                />
            {:else}
                <p class="error">{result.message}</p>
            {/if}
            
            <DebugInfo info={result.debugInfo} />
        </div>
    {/if}
</main>

<style>
    main {
        max-width: 800px;
        margin: 0 auto;
        padding: var(--spacing-8);
    }
    
    h2 {
        margin-bottom: var(--spacing-2);
        color: var(--color-text);
        font-size: var(--font-size-2xl);
        text-align: center;
    }

    .page-description {
        text-align: center;
        color: var(--color-text-light);
        margin-bottom: var(--spacing-8);
    }
    
    .tts-form {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-6);
        margin-bottom: var(--spacing-8);
        background: var(--color-surface);
        padding: var(--spacing-6);
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow-base);
    }
    
    .form-group {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-2);
    }
    
    label {
        font-weight: 500;
        color: var(--color-text);
        font-size: var(--font-size-sm);
    }
    
    textarea, input {
        padding: var(--spacing-3);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-base);
        font-size: var(--font-size-base);
        background: var(--color-background);
        transition: all var(--transition-base);
    }
    
    textarea:focus, input:focus {
        outline: none;
        border-color: var(--color-primary);
        box-shadow: var(--shadow-sm);
    }
    
    button {
        padding: var(--spacing-3) var(--spacing-6);
        background: var(--color-primary);
        color: var(--color-surface);
        border: none;
        border-radius: var(--border-radius-base);
        font-size: var(--font-size-base);
        font-weight: 500;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: var(--spacing-2);
        transition: all var(--transition-base);
        box-shadow: var(--shadow-sm);
    }
    
    button:disabled {
        opacity: 0.7;
        cursor: not-allowed;
        transform: none;
    }
    
    button:not(:disabled):hover {
        background: var(--color-primary-dark);
        transform: translateY(-1px);
        box-shadow: var(--shadow-base);
    }
    
    .result {
        margin-top: var(--spacing-8);
        background: var(--color-surface);
        padding: var(--spacing-6);
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow-base);
    }
    
    .error {
        color: var(--color-error);
        padding: var(--spacing-4);
        background: #fef2f2;
        border: 1px solid #fee2e2;
        border-radius: var(--border-radius-base);
        margin-bottom: var(--spacing-4);
    }

    @media (max-width: 640px) {
        main {
            padding: var(--spacing-4);
        }

        h2 {
            font-size: var(--font-size-xl);
            margin-bottom: var(--spacing-2);
        }

        .page-description {
            font-size: var(--font-size-sm);
            margin-bottom: var(--spacing-6);
        }

        .tts-form {
            padding: var(--spacing-4);
            gap: var(--spacing-4);
        }

        .result {
            padding: var(--spacing-4);
        }
    }
</style>
