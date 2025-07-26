import { StyleSheet } from 'react-native'

// Color constants
export const Colors = {
  primary: '#007AFF',
  background: '#F2F2F7',
  white: '#FFFFFF',
  black: '#000000',
  text: '#000000',
  textSecondary: '#666666',
  border: '#E0E0E0',
  error: '#FF3B30',
  disabled: '#C0C0C0',
  inputBackground: '#F8F8F8',
  shadow: '#000000',
} as const

// Common spacing values
export const Spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
} as const

// Common font sizes
export const FontSizes = {
  small: 12,
  medium: 14,
  large: 16,
  xlarge: 18,
  xxlarge: 24,
} as const

// Shared styles
export const CommonStyles = StyleSheet.create({
  // Container styles
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  setupContainer: {
    flex: 1,
    padding: Spacing.lg,
  },
  scrollContent: {
    paddingBottom: 20,
  },

  // Header styles
  header: {
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    backgroundColor: Colors.white,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: FontSizes.xlarge,
    fontWeight: '600',
    color: Colors.text,
    textAlign: 'center',
  },
  headerButton: {
    marginRight: 4,
  },
  headerButtonText: {
    color: Colors.primary,
    fontSize: FontSizes.large,
    fontWeight: '500',
  },

  // Text styles
  setupTitle: {
    fontSize: FontSizes.xxlarge,
    fontWeight: 'bold',
    color: Colors.text,
    marginBottom: Spacing.sm,
    textAlign: 'center',
  },
  setupDescription: {
    fontSize: FontSizes.large,
    color: Colors.textSecondary,
    lineHeight: 24,
    marginBottom: Spacing.xxl,
    textAlign: 'center',
  },
  description: {
    fontSize: FontSizes.medium,
    color: Colors.textSecondary,
    lineHeight: 20,
    marginVertical: Spacing.lg,
    textAlign: 'center',
  },

  // Button styles
  button: {
    margin: 10,
    padding: 10,
    backgroundColor: '#333',
    borderRadius: 5,
  },
  buttonText: {
    color: Colors.white,
    fontSize: FontSizes.large,
  },
  primaryButton: {
    backgroundColor: Colors.primary,
    paddingHorizontal: Spacing.xl,
    paddingVertical: Spacing.md,
    borderRadius: Spacing.sm,
  },
  primaryButtonText: {
    color: Colors.white,
    fontSize: FontSizes.large,
    fontWeight: '600',
  },
  primaryButtonDisabled: {
    backgroundColor: Colors.disabled,
  },
  secondaryButton: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    backgroundColor: Colors.error,
    borderRadius: 6,
  },
  secondaryButtonText: {
    color: Colors.white,
    fontSize: FontSizes.medium,
    fontWeight: '500',
  },
  textButton: {
    fontSize: FontSizes.large,
    color: Colors.primary,
    fontWeight: '500',
  },
  disabledButton: {
    opacity: 0.5,
  },

  // Loading styles
  loadingContainer: {
    alignItems: 'center',
    marginTop: Spacing.xxl,
  },
  loadingText: {
    marginTop: Spacing.sm,
    fontSize: FontSizes.large,
    color: Colors.textSecondary,
  },

  // Progress bar styles
  progressContainer: {
    marginTop: Spacing.lg,
    width: '100%',
    alignItems: 'center',
  },
  progressBar: {
    width: '80%',
    height: 8,
    backgroundColor: Colors.border,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: Colors.primary,
    borderRadius: 4,
  },

  // Card styles
  card: {
    backgroundColor: Colors.white,
    borderRadius: Spacing.md,
    padding: Spacing.lg,
    marginVertical: Spacing.sm,
    marginHorizontal: Spacing.lg,
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },

  // Form styles
  paramGroup: {
    backgroundColor: Colors.white,
    borderRadius: Spacing.md,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
  },
  paramLabel: {
    fontSize: FontSizes.large,
    fontWeight: '600',
    color: Colors.text,
    marginBottom: 4,
  },
  paramDescription: {
    fontSize: FontSizes.medium,
    color: Colors.textSecondary,
    lineHeight: 18,
    marginBottom: Spacing.md,
  },
  textInput: {
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: Spacing.sm,
    paddingHorizontal: Spacing.md,
    paddingVertical: 10,
    fontSize: FontSizes.large,
    backgroundColor: Colors.inputBackground,
  },

  // Modal styles
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: Colors.white,
    borderRadius: Spacing.lg,
    padding: Spacing.xl,
    margin: Spacing.xl,
    maxHeight: '80%',
    maxWidth: '95%',
    minWidth: '85%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    backgroundColor: Colors.white,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  modalTitle: {
    fontSize: FontSizes.xlarge,
    fontWeight: '600',
    color: Colors.text,
  },
  modalCloseButton: {
    fontSize: FontSizes.xlarge,
    color: Colors.primary,
    fontWeight: '600',
  },

  // Settings styles
  settingsContainer: {
    alignItems: 'center',
    marginVertical: Spacing.lg,
  },

  // Utility styles
  flexRow: {
    flexDirection: 'row',
  },
  flexColumn: {
    flexDirection: 'column',
  },
  alignCenter: {
    alignItems: 'center',
  },
  justifyCenter: {
    justifyContent: 'center',
  },
  justifyBetween: {
    justifyContent: 'space-between',
  },
  flex1: {
    flex: 1,
  },
  marginBottom: {
    marginBottom: Spacing.lg,
  },
  marginTop: {
    marginTop: Spacing.lg,
  },
})
