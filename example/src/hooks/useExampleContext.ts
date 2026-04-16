import { useCallback, useEffect, useRef, useState } from 'react'
import type { LlamaContext } from '../../../src'

export function useExampleContext() {
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [initProgress, setInitProgress] = useState(0)
  const contextRef = useRef<LlamaContext | null>(null)

  useEffect(() => {
    contextRef.current = context
  }, [context])

  useEffect(
    () => () => {
      if (contextRef.current) {
        void contextRef.current.release().catch((error) => {
          console.warn('Failed to release llama context during cleanup:', error)
        })
      }
    },
    [],
  )

  const releaseContext = useCallback(async () => {
    if (!contextRef.current) {
      setContext(null)
      setIsModelReady(false)
      return
    }

    const currentContext = contextRef.current
    contextRef.current = null
    setContext(null)
    setIsModelReady(false)
    await currentContext.release()
  }, [])

  const replaceContext = useCallback(async (nextContext: LlamaContext) => {
    if (contextRef.current && contextRef.current !== nextContext) {
      await contextRef.current.release()
    }

    contextRef.current = nextContext
    setContext(nextContext)
    setIsModelReady(true)
  }, [])

  return {
    context,
    contextRef,
    initProgress,
    isModelReady,
    setContext,
    setInitProgress,
    setIsModelReady,
    releaseContext,
    replaceContext,
  }
}
