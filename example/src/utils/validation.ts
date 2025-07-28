export const validateNumber = (text: string, min = 0, max = Infinity): number | undefined => {
  const num = parseFloat(text)
  return !Number.isNaN(num) && num >= min && num <= max ? num : undefined
}

export const validateInteger = (text: string, min = 0, max = Infinity): number | undefined => {
  const num = parseInt(text, 10)
  return !Number.isNaN(num) && num >= min && num <= max ? num : undefined
}
